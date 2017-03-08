#pragma once

#include <chrono>
#include <mutex>
#include <condition_variable>

#include "core/info.hpp"
#include "husky/base/exception.hpp"
#include "husky/base/serialization.hpp"
#include "husky/core/zmq_helpers.hpp"

#include "ml/mlworker/mlworker.hpp"
#include "ml/consistency/consistency_controller.hpp"
#include "ml/consistency/asp_consistency_controller.hpp"
#include "ml/consistency/ssp_consistency_controller.hpp"
#include "ml/consistency/bsp_consistency_controller.hpp"
#include "ml/model/integral_model.hpp"
#include "ml/model/chunk_based_mt_model.hpp"
#include "ml/shared/shared_state.hpp"

#include "kvstore/kvstore.hpp"

#include "core/color.hpp"

namespace ml {
namespace mlworker {


class SPMTWorker : public mlworker::GenericMLWorker {
    /*
     * The shared state needed by SPMT
     */
    struct SPMTState {
        // pointer to consistency controller
        consistency::AbstractConsistencyController* p_controller_;
        // pointer to model
        model::Model* p_model_;
    };
   public:
    SPMTWorker() = delete;
    SPMTWorker(const SPMTWorker&) = delete;
    SPMTWorker& operator=(const SPMTWorker&) = delete;
    SPMTWorker(SPMTWorker&&) = delete;
    SPMTWorker& operator=(SPMTWorker&&) = delete;

    /*
     * @param type: 0 for ssp, 1 for bsp
     */
    SPMTWorker(const husky::Info& info, zmq::context_t& context, bool is_hogwild = false)
        : shared_state_(info.get_task_id(), info.is_leader(), info.get_num_local_workers(), context),
          info_(info),
          is_hogwild_(is_hogwild) {
        int model_id = static_cast<husky::MLTask*>(info_.get_task())->get_kvstore();
        size_t num_params = static_cast<husky::MLTask*>(info_.get_task())->get_dimensions();
        // check valid
        if (info_.get_num_local_workers() != info_.get_num_workers()) {
            throw husky::base::HuskyException("[SPMT] threads are not in the same machine. Task is:" +
                                              std::to_string(info.get_task_id()));
        }

        // Find flags from hint
        auto& hint = info.get_task()->get_hint();
        if (hint.find(husky::constants::kEnableDirectModelTransfer) != hint.end()) {
            enable_direct_model_transfer_ = true;
        }
        if (hint.find(husky::constants::kParamType) != hint.end() 
                && hint.at(husky::constants::kParamType) == husky::constants::kChunkType) {
            use_chunk_model_ = true;
            assert(enable_direct_model_transfer_ == false);
        } else {
            use_chunk_model_ = false;
        }

        if (info_.is_leader() == true) {
            SPMTState* state = new SPMTState;
            if (is_hogwild_ == false) {  // if it's not hogwild, set consistency
                // Set controller
                std::string type = info.get_task()->get_hint().at(husky::constants::kConsistency);
                if (type == husky::constants::kBSP) {
                    state->p_controller_  = new consistency::BSPConsistencyController;
                } else if (type == husky::constants::kSSP) {
                    state->p_controller_ = new consistency::SSPConsistencyController;
                } else if (type == husky::constants::kASP) {
                    state->p_controller_ = new consistency::ASPConsistencyController;
                } else {
                    assert(false);
                }
                state->p_controller_->Init(info.get_num_local_workers());
            } else {
                state->p_controller_ = nullptr;
            }

            // Set chunk or integral
            if (use_chunk_model_ == true) {
                // Use Chunk model
                if (is_hogwild_) {
                    state->p_model_ = (model::Model*) new model::ChunkBasedMTModel(model_id, num_params);
                } else {
                    state->p_model_ = (model::Model*) new model::ChunkBasedMTLockModel(model_id, num_params);
                }
            } else {
                // Use Integral model
                state->p_model_ = (model::Model*) new model::IntegralModel(model_id, num_params);
            }
            // 1. Init shared_state_
            shared_state_.Init(state);
        }
        // 2. Sync shared_state_
        shared_state_.SyncState();
        // 3. Load
        if (use_chunk_model_ == false)
            Load();

        // For logging debug message
        if (info.is_leader() == true) {
            std::string model_type = use_chunk_model_ ? "ChunkBasedModel" : "IntegralModel";
            std::string consistency = is_hogwild_ ? "hogwild ASP" : info.get_task()->get_hint().at(husky::constants::kConsistency);
            husky::LOG_I << CLAY("[SPMT] model_id: "+std::to_string(model_id)
                    +"; local_id: "+std::to_string(info.get_local_id())
                    +"; model_size: "+std::to_string(num_params)
                    +"; model_type: "+model_type
                    +"; direct_model_transfer: "+std::to_string(enable_direct_model_transfer_)
                    +"; consistency: "+consistency);
        }
    }
    ~SPMTWorker() {
        Dump();
        shared_state_.Barrier();
        if (info_.is_leader() == true) {
            if (is_hogwild_ == false) {
                delete shared_state_.Get()->p_controller_;
            }
            delete shared_state_.Get()->p_model_;
            delete shared_state_.Get();
        }
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        shared_state_.Get()->p_controller_->BeforePush(info_.get_cluster_id());
        shared_state_.Get()->p_model_->Push(keys, vals);
        shared_state_.Get()->p_controller_->AfterPush(info_.get_cluster_id());
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals) override {
        shared_state_.Get()->p_controller_->BeforePull(info_.get_cluster_id());
        shared_state_.Get()->p_model_->Pull(keys, vals, info_.get_local_id());
        shared_state_.Get()->p_controller_->AfterPull(info_.get_cluster_id());
    }

    // For v2
    // TODO: Now, the v2 APIs for spmt still need copy,
    // Later, we may use brunching to facilitate zero-copy when doing single/hogwild
    virtual void Prepare_v2(const std::vector<husky::constants::Key>& keys) override {
        keys_ = const_cast<std::vector<husky::constants::Key>*>(&keys);
        Pull(keys, &vals_);
        delta_.clear();
        delta_.resize(keys.size());
    }
    virtual float Get_v2(husky::constants::Key idx) override { return vals_[idx]; }
    virtual void Update_v2(husky::constants::Key idx, float val) override {
        delta_[idx] += val;
        vals_[idx] += val;
    }
    virtual void Update_v2(const std::vector<float>& vals) override {
        assert(vals.size() == vals_.size());
        for (size_t i = 0; i < vals.size(); ++i) {
            vals_[i] += vals[i];
            delta_[i] += vals[i];
        }
    }
    virtual void Clock_v2() override { Push(*keys_, delta_); }

    /*
     * Load the model from kvstore/direct_model_transfer
     */
    void Load() {
        if (info_.is_leader() == true) {
            // hint will be set to kTransfer if enable_direct_model_transfer_ and it's not the first epoch
            std::string hint = (enable_direct_model_transfer_ == true && info_.get_current_epoch() != 0) ? husky::constants::kTransfer : husky::constants::kKVStore;
            shared_state_.Get()->p_model_->Load(info_.get_local_id(), hint);
        }
        shared_state_.Barrier();
    }

    /*
     * Dump the model to kvstore/direct_model_transfer
     */
    void Dump() {
        shared_state_.Barrier();
        if (info_.is_leader() == true) {
            // hint will be set to kTransfer if enable_direct_model_transfer_ and it's not the last epoch
            std::string hint = (enable_direct_model_transfer_ == true && info_.get_current_epoch() < info_.get_total_epoch()-1) ? husky::constants::kTransfer : husky::constants::kKVStore;
            shared_state_.Get()->p_model_->Dump(info_.get_local_id(), hint);
        }
        shared_state_.Barrier();
    }

   protected:
    bool is_hogwild_ = false;
    const husky::Info& info_;
    SharedState<SPMTState> shared_state_;
    bool use_chunk_model_ = false;

    // For v2
    // Pointer to keys
    std::vector<husky::constants::Key>* keys_;

   private:
    bool enable_direct_model_transfer_ = false;

    // For v2
    std::vector<float> vals_;
    std::vector<float> delta_;
};

}  // namespace mlworker
}  // namespace ml
