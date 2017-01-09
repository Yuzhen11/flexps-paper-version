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
    virtual void Push(const std::vector<int>& keys, const std::vector<float>& vals) override {
        assert(push_count_ + 1 == pull_count_);
        push_count_ += 1;
        ts_ = kvworker_->Push(model_id_, keys, vals, nullptr);
    }
    virtual void Pull(const std::vector<int>& keys, std::vector<float>* vals) override {
        assert(push_count_ == pull_count_);
        pull_count_ += 1;
        if (ts_ != -1)
            kvworker_->Wait(model_id_, ts_);  // Wait for last Push
        ts_ = kvworker_->Pull(model_id_, keys, vals, nullptr);
        kvworker_->Wait(model_id_, ts_);  // Wait for this Pull
    }
   private:
    int model_id_;
    kvstore::KVWorker* kvworker_ = nullptr;

    // Just to restrict the usage of the Push/Pull APIs,
    // The correct usage should be Pull, Push, Pull, Push...
    int push_count_ = 0;
    int pull_count_ = 0;
    int ts_ = -1;
};

}  // namespace ps
}  // namespace ml
