#pragma once

#include <chrono>

#include <utility>

#include "ml/common/mlworker.hpp"
#include "ml/model/load.hpp"
#include "ml/model/dump.hpp"
#include "ml/model/integral_model_with_ptr.hpp"
#include "ml/model/chunk_based_model_with_ptr.hpp"

#include "kvstore/kvstore.hpp"
#include "kvstore/kvstore.hpp"


namespace ml {
namespace single {

class SingleGenericWorker : public common::GenericMLWorker {
   public:
    SingleGenericWorker() = delete;
    SingleGenericWorker(const SingleGenericWorker&) = delete;
    SingleGenericWorker& operator=(const SingleGenericWorker&) = delete;
    SingleGenericWorker(SingleGenericWorker&&) = delete;
    SingleGenericWorker& operator=(SingleGenericWorker&&) = delete;

    SingleGenericWorker(const husky::Info& info)
        : info_(info) {
        size_t num_params = static_cast<husky::MLTask*>(info_.get_task())->get_dimensions();
        int model_id = static_cast<husky::MLTask*>(info_.get_task())->get_kvstore();

        auto& hint = info.get_task()->get_hint();
        if (hint.find(husky::constants::kEnableDirectModelTransfer) != hint.end()) {
            enable_direct_model_transfer_ = true;
        }
        if (hint.find(husky::constants::kParamType) != hint.end() 
                && hint.at(husky::constants::kParamType) == husky::constants::kChunkType) {
            assert(enable_direct_model_transfer_ == false);
            // Use Chunk model
            model_.reset(new model::ChunkBasedModelWithPtr(model_id, num_params));
            p_chunk_params_ = static_cast<model::ChunkBasedModelWithPtr*>(model_.get())->GetParamsPtr();
            use_chunk_model_ = true;
            chunk_size_ = kvstore::RangeManager::Get().GetChunkSize(model_id);
        } else {
            // Use Integral model
            model_.reset(new model::IntegralModelWithPtr(model_id, num_params));
            p_integral_params_= static_cast<model::IntegralModelWithPtr*>(model_.get())->GetParamsPtr();
            use_chunk_model_ = false;
            // Load 
            Load();
        }

        // For logging debug message
        std::string model_type = use_chunk_model_ ? "ChunkBasedModel" : "IntegralModel";
        husky::LOG_I << CLAY("[Single] model_id: "+std::to_string(model_id)
                +"; local_id: "+std::to_string(info_.get_local_id())
                +"; model_size: "+std::to_string(num_params)
                +"; model_type: "+model_type
                +"; direct_model_transfer: "+std::to_string(enable_direct_model_transfer_));
    }

    virtual void Load() override {
        // hint will be set to kTransfer if enable_direct_model_transfer_ and it's not the first epoch
        std::string hint = (enable_direct_model_transfer_ == true && info_.get_current_epoch() != 0) ? husky::constants::kTransfer : husky::constants::kKVStore;
        model_->Load(info_.get_local_id(), hint);
    }

    virtual void Dump() override {
        // hint will be set to kTransfer if enable_direct_model_transfer_ and it's not the last epoch
        std::string hint = (enable_direct_model_transfer_ == true && info_.get_current_epoch() < info_.get_total_epoch()-1) ? husky::constants::kTransfer : husky::constants::kKVStore;
        model_->Dump(info_.get_local_id(), hint);
    }

    /*
     * Put/Get, Push/Pull APIs
     */
    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        model_->Push(keys, vals);
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals) override {
        model_->Pull(keys, vals, info_.get_local_id());
    }

    // For v2
    virtual void Prepare_v2(const std::vector<husky::constants::Key>& keys) override {
        keys_ = const_cast<std::vector<husky::constants::Key>*>(&keys);
        if (!p_integral_params_)
            static_cast<model::ChunkBasedModelWithPtr*>(model_.get())->Prepare(keys, info_.get_local_id());
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
    std::unique_ptr<model::Model> model_;
    // A pointer ponints to the parameter directly
    std::vector<float>* p_integral_params_ = nullptr;
    std::vector<std::vector<float>>* p_chunk_params_ = nullptr;
    int chunk_size_ = -1;  // Only for ChunkBasedModel
    
    const husky::Info& info_;

    bool enable_direct_model_transfer_ = false;
    bool use_chunk_model_ = false;

    // For v2
    // Pointer to keys
    std::vector<husky::constants::Key>* keys_;
};

}  // namespace single
}  // namespace ml
