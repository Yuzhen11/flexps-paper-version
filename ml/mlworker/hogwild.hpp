#pragma once

#include <chrono>

#include "core/info.hpp"
#include "husky/base/exception.hpp"
#include "husky/base/serialization.hpp"
#include "husky/core/zmq_helpers.hpp"

#include "ml/mlworker/mlworker.hpp"
#include "ml/model/load.hpp"
#include "ml/model/dump.hpp"
#include "ml/shared/shared_state.hpp"
#include "ml/model/integral_model.hpp"
#include "ml/model/chunk_based_mt_model.hpp"

#include "kvstore/kvstore.hpp"

#include "core/color.hpp"

namespace ml {
namespace mlworker {


/*
 * For the HogwildWorker, the type ModelType is now fixed to std::vector<float>
 */
class HogwildWorker : public mlworker::GenericMLWorker {
    /*
     * The shared state needed by Hogwild
     */
    struct HogwildState {
        model::Model* p_model_;
    };
   public:
    HogwildWorker() = delete;
    HogwildWorker(const HogwildWorker&) = delete;
    HogwildWorker& operator=(const HogwildWorker&) = delete;
    HogwildWorker(HogwildWorker&&) = delete;
    HogwildWorker& operator=(HogwildWorker&&) = delete;

    /*
     * constructor to construct a hogwild model
     * \param context zmq_context
     * \param info info in this instance
     */
    HogwildWorker(const husky::Info& info, zmq::context_t& context)
        : shared_state_(info.get_task_id(), info.is_leader(), info.get_num_local_workers(), context),
          info_(info) {
        int model_id = static_cast<husky::MLTask*>(info.get_task())->get_kvstore();
        size_t num_params = static_cast<husky::MLTask*>(info_.get_task())->get_dimensions();
        // check valid
        if (!isValid()) {
            throw husky::base::HuskyException("[Hogwild] threads are not in the same machine. Task is:" +
                                              std::to_string(info.get_task_id()));
        }
        husky::LOG_I << info.is_leader() << " " << info.get_cluster_id() << std::endl;

        // Find flags from hint
        auto& hint = info.get_task()->get_hint();
        if (hint.find(husky::constants::kEnableDirectModelTransfer) != hint.end()) {
            enable_direct_model_transfer_ = true;
        }
        if (hint.find(husky::constants::kParamType) != hint.end() 
                && hint.at(husky::constants::kParamType) == husky::constants::kChunkType) {
            use_chunk_model_ = true;
        } else {
            use_chunk_model_ = false;
        }

        if (info_.is_leader() == true) {
            HogwildState* state = new HogwildState;
            if (use_chunk_model_ == true) {
                // Use Chunk model
                state->p_model_ = (model::Model*) new model::ChunkBasedMTModel(model_id, num_params);
                chunk_size_ = kvstore::RangeManager::Get().GetChunkSize(model_id);
            } else {
                // Use Integral model
                state->p_model_ = (model::Model*) new model::IntegralModel(model_id, num_params);
            }
            // 1. Init shared_state_
            shared_state_.Init(state);
        }

        // 2. Sync shared_state_
        shared_state_.SyncState();
        if (use_chunk_model_ == true) {
            p_chunk_params_ = static_cast<model::ChunkBasedMTModel*>(shared_state_.Get()->p_model_)->GetParamsPtr();
        } else {
            p_integral_params_ = static_cast<model::IntegralModel*>(shared_state_.Get()->p_model_)->GetParamsPtr();
            // 3. Load
            Load();
        }

        // For logging debug message
        std::string model_type = use_chunk_model_ ? "ChunkBasedModel" : "IntegralModel";
        if (info.is_leader() == true) {
            husky::LOG_I << CLAY("[Hogwild] model_id: "+std::to_string(model_id)
                    +"; local_id: "+std::to_string(info.get_local_id())
                    +"; model_size: "+std::to_string(num_params)
                    +"; model_type: "+model_type
                    +"; direct_model_transfer: "+std::to_string(enable_direct_model_transfer_));
        }

    }

    /*
     * destructor
     * 1. Sync() and 2. leader delete the model
     */
    ~HogwildWorker() {
        shared_state_.Barrier();
        if (info_.is_leader() == true) {
            delete shared_state_.Get()->p_model_;
            delete shared_state_.Get();
        }
    }

    /*
     * Get parameters from global kvstore
     */
    virtual void Load() override {
        if (info_.is_leader() == true) {
            // hint will be set to kTransfer if enable_direct_model_transfer_ and it's not the first epoch
            std::string hint = (enable_direct_model_transfer_ == true && info_.get_current_epoch() != 0) ? husky::constants::kTransfer : husky::constants::kKVStore;
            shared_state_.Get()->p_model_->Load(info_.get_local_id(), hint);
        }
        // Other threads should wait
        shared_state_.Barrier();
    }
    /*
     * Put the parameters to global kvstore
     */
    virtual void Dump() override {
        shared_state_.Barrier();
        if (info_.is_leader() == true) {
            // hint will be set to kTransfer if enable_direct_model_transfer_ and it's not the last epoch
            std::string hint = (enable_direct_model_transfer_ == true && info_.get_current_epoch() < info_.get_total_epoch()-1) ? husky::constants::kTransfer : husky::constants::kKVStore;
            shared_state_.Get()->p_model_->Dump(info_.get_local_id(), hint);
        }
        shared_state_.Barrier();
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        shared_state_.Get()->p_model_->Push(keys, vals);
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals) override {
        shared_state_.Get()->p_model_->Pull(keys, vals, info_.get_local_id());
    }

    // For v2
    virtual void Prepare_v2(const std::vector<husky::constants::Key>& keys) override {
        keys_ = const_cast<std::vector<husky::constants::Key>*>(&keys);
        if (!p_integral_params_)
            static_cast<model::ChunkBasedMTModel*>(shared_state_.Get()->p_model_)->Prepare(keys, info_.get_local_id());
    }
    virtual float Get_v2(size_t idx) override { 
        if (p_integral_params_)
            return (*p_integral_params_)[(*keys_)[idx]];
        else
            return (*p_chunk_params_)[(*keys_)[idx]/chunk_size_][(*keys_)[idx]%chunk_size_];
    }
    virtual void Update_v2(size_t idx, float val) override { 
        if (p_integral_params_)
            (*p_integral_params_)[(*keys_)[idx]] += val;
        else
            (*p_chunk_params_)[(*keys_)[idx]/chunk_size_][(*keys_)[idx]%chunk_size_] += val;
    }

   private:
    /*
     * Serve as a barrier
     */
    virtual void Sync() override {
        shared_state_.Barrier();
    }

    /*
     * check whether all the threads are in the same machine
     */
    bool isValid() {
        // husky::base::log_msg("locals: " + std::to_string(info_.get_num_local_workers()) + " globals:" +
        //                      std::to_string(info_.get_num_workers()));
        return info_.get_num_local_workers() == info_.get_num_workers();
    }

    const husky::Info& info_;
    SharedState<HogwildState> shared_state_;
    // A pointer points to the parameter directly
    std::vector<float>* p_integral_params_ = nullptr;
    std::vector<std::vector<float>>* p_chunk_params_ = nullptr;
    int chunk_size_ = -1;  // Only for ChunkBasedModel

    bool enable_direct_model_transfer_ = false;
    bool use_chunk_model_ = false;

    // For v2
    // Pointer to keys
    std::vector<husky::constants::Key>* keys_;
};

}  // namespace mlworker
}  // namespace ml
