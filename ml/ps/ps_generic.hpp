#pragma once

#include "kvstore/kvstore.hpp"

namespace ml {
namespace ps {

/*
 * Generic model for PS
 *
 * Just a wraper for kvstore::KVWorker, so users doesn't need to care about wait and timestamp
 *
 * ASP/BSP/SSP may all use this worker
 * SSP may have other workers 
 *
 * Assume that in each epoch, Pull will be invoked first and then the Push.
 */
class PSGenericWorker : public common::GenericMLWorker {
   public:
    PSGenericWorker() = delete;
    PSGenericWorker(int model_id, int local_id): model_id_(model_id),
        kvworker_(kvstore::KVStore::Get().get_kvworker(local_id)) {
    }
    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        assert(push_count_ + 1 == pull_count_);
        push_count_ += 1;
        ts_ = kvworker_->Push(model_id_, keys, vals, nullptr);
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals) override {
        assert(push_count_ == pull_count_);
        pull_count_ += 1;
        if (ts_ != -1)
            kvworker_->Wait(model_id_, ts_);  // Wait for last Push
        ts_ = kvworker_->Pull(model_id_, keys, vals, nullptr);
        kvworker_->Wait(model_id_, ts_);  // Wait for this Pull
    }
    
    // For v2
    virtual void Prepare_v2(std::vector<husky::constants::Key>& keys) override {
        keys_ = &keys;
        Pull(keys, &vals_);
        delta_.clear();
        delta_.resize(keys.size());
    }
    virtual float Get_v2(husky::constants::Key idx) override {
        return vals_[idx];
    }
    virtual void Update_v2(husky::constants::Key idx, float val) override {
        delta_[idx] += val;
        vals_[idx] += val;
    }
    virtual void Update_v2(const std::vector<float>& vals) override {
        assert(vals.size() == vals_.size());
        for (size_t i = 0; i < vals.size(); ++ i) {
            vals_[i] += vals[i];
            delta_[i] += vals[i];
        }
    }
    virtual void Clock_v2() override {
        Push(*keys_, delta_);
    }
    
   private:
    int model_id_;
    kvstore::KVWorker* kvworker_ = nullptr;

    // Just to restrict the usage of the Push/Pull APIs,
    // The correct usage should be Pull, Push, Pull, Push...
    int push_count_ = 0;
    int pull_count_ = 0;
    int ts_ = -1;

    // For v2
    // Pointer to keys
    std::vector<husky::constants::Key>* keys_;
    std::vector<float> vals_;
    std::vector<float> delta_;
};

}  // namespace ps
}  // namespace ml
