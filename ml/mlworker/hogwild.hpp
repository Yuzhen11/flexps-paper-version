#pragma once

#include "ml/mlworker/spmt.hpp"

namespace ml {
namespace mlworker {


/*
 * HogwildWorker is a special case of SPMTWorker
 */
class HogwildWorker : public SPMTWorker {
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
        : SPMTWorker(info, context, true) {
        if (use_chunk_model_ == true) {
            int model_id = static_cast<husky::MLTask*>(info_.get_task())->get_kvstore();
            chunk_size_ = kvstore::RangeManager::Get().GetChunkSize(model_id);
            p_chunk_params_ = static_cast<model::ChunkBasedMTModel*>(shared_state_.Get()->p_model_)->GetParamsPtr();
        } else {
            p_integral_params_ = static_cast<model::IntegralModel*>(shared_state_.Get()->p_model_)->GetParamsPtr();
        }
    }

    ~HogwildWorker() {
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
    virtual void Update_v2(const std::vector<float>& vals) override {
        throw husky::base::HuskyException("Not implemented in Hogwild");
    }
    virtual void Clock_v2() override {}

   private:
    // A pointer points to the parameter directly
    std::vector<float>* p_integral_params_ = nullptr;
    std::vector<std::vector<float>>* p_chunk_params_ = nullptr;
    int chunk_size_ = -1;  // Only for ChunkBasedModel
};

}  // namespace mlworker
}  // namespace ml
