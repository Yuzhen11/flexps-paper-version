#pragma once

#include <chrono>

#include <utility>

#include "ml/mlworker/mlworker.hpp"
#include "ml/model/load.hpp"
#include "ml/model/dump.hpp"
#include "ml/model/integral_model.hpp"
#include "ml/model/chunk_based_model.hpp"

#include "kvstore/kvstore.hpp"
#include "kvstore/kvstore.hpp"


namespace ml {
namespace mlworker {

template<typename Val>
class SingleWorker : public mlworker::GenericMLWorker<Val> {
   public:
    SingleWorker() = delete;
    SingleWorker(const SingleWorker&) = delete;
    SingleWorker& operator=(const SingleWorker&) = delete;
    SingleWorker(SingleWorker&&) = delete;
    SingleWorker& operator=(SingleWorker&&) = delete;

    SingleWorker(const husky::Info& info, const husky::TableInfo& table_info)
        : info_(info) {
        size_t num_params = table_info.dims;
        int model_id = table_info.kv_id;

        enable_direct_model_transfer_ = table_info.kEnableDirectModelTransfer;
        if (table_info.param_type == husky::ParamType::ChunkType) {
            assert(enable_direct_model_transfer_ == false);
            // Use Chunk model
            model_.reset(new model::ChunkBasedModel<Val>(model_id, num_params));
            p_chunk_params_ = static_cast<model::ChunkBasedModel<Val>*>(model_.get())->GetParamsPtr();
            use_chunk_model_ = true;
            chunk_size_ = kvstore::RangeManager::Get().GetChunkSize(model_id);
        } else if (table_info.param_type == husky::ParamType::IntegralType) {
            // Use Integral model
            model_.reset(new model::IntegralModel<Val>(model_id, num_params));
            p_integral_params_= static_cast<model::IntegralModel<Val>*>(model_.get())->GetParamsPtr();
            use_chunk_model_ = false;
            // Load 
            Load();
        } else {
            husky::LOG_I << "table_info: " << table_info.DebugString();
            assert(false);
        }

        // For logging debug message
        std::string model_type = use_chunk_model_ ? "ChunkBasedModel" : "IntegralModel";
        husky::LOG_I << CLAY("[Single] model_id: "+std::to_string(model_id)
                +"; local_id: "+std::to_string(info_.get_local_id())
                +"; model_size: "+std::to_string(num_params)
                +"; model_type: "+model_type
                +"; direct_model_transfer: "+std::to_string(enable_direct_model_transfer_));
    }

    ~SingleWorker() {
        Dump();
    }

    void Load() {
        // hint will be set to kTransfer if enable_direct_model_transfer_ and it's not the first epoch
        std::string hint = (enable_direct_model_transfer_ == true && info_.get_current_epoch() != 0) ? husky::constants::kTransferIntegral : husky::constants::kKVStoreIntegral;
        model_->Load(info_.get_local_id(), info_.get_task()->get_id(), hint);
    }

    void Dump() {
        // hint will be set to kTransfer if enable_direct_model_transfer_ and it's not the last epoch
        std::string hint = (enable_direct_model_transfer_ == true && info_.get_current_epoch() < info_.get_total_epoch()-1) ? husky::constants::kTransferIntegral : husky::constants::kKVStoreIntegral;
        model_->Dump(info_.get_local_id(), info_.get_task()->get_id(), hint);
    }

    /*
     * Put/Get, Push/Pull APIs
     */
    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) override {
        model_->Push(keys, vals);
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals) override {
        model_->Pull(keys, vals, info_.get_local_id());
    }

    // For v2
    virtual void Prepare_v2(const std::vector<husky::constants::Key>& keys) override {
        keys_ = const_cast<std::vector<husky::constants::Key>*>(&keys);
        if (!p_integral_params_)
            static_cast<model::ChunkBasedModel<Val>*>(model_.get())->Prepare(keys, info_.get_local_id());
    }
    virtual Val Get_v2(size_t idx) override { 
        if (p_integral_params_)
            return (*p_integral_params_)[(*keys_)[idx]]; 
        else
            return (*p_chunk_params_)[(*keys_)[idx]/chunk_size_][(*keys_)[idx]%chunk_size_];
    }
    virtual void Update_v2(size_t idx, Val val) override { 
        if (p_integral_params_)
            (*p_integral_params_)[(*keys_)[idx]] += val; 
        else
            (*p_chunk_params_)[(*keys_)[idx]/chunk_size_][(*keys_)[idx]%chunk_size_] += val;
    }

   private:
    std::unique_ptr<model::Model<Val>> model_;
    // A pointer ponints to the parameter directly
    std::vector<Val>* p_integral_params_ = nullptr;
    std::vector<std::vector<Val>>* p_chunk_params_ = nullptr;
    int chunk_size_ = -1;  // Only for ChunkBasedModel
    
    const husky::Info& info_;

    bool enable_direct_model_transfer_ = false;
    bool use_chunk_model_ = false;

    // For v2
    // Pointer to keys
    std::vector<husky::constants::Key>* keys_;
};

}  // namespace mlworker
}  // namespace ml
