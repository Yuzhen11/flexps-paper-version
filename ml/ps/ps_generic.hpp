#pragma once

#include "kvstore/kvstore.hpp"

namespace ml {
namespace ps {

class PSGenericModel : public common::GenericMLWorker {
   public:
    PSGenericModel() = delete;
    PSGenericModel(int model_id, int local_id): model_id_(model_id), local_id_(local_id),
        kvworker_(kvstore::KVStore::Get().get_kvworker(local_id)) {
    }

    virtual void Sync() override {
        for (auto i : unfinished_timestamps_) {
            Wait(i);
        }
        unfinished_timestamps_.clear();
    };

    virtual int Push(const std::vector<int>& keys, const std::vector<float>& vals, const Callback& cb = nullptr) override {
        int ts = kvworker_->Push(model_id_, keys, vals, cb);
        unfinished_timestamps_.insert(ts);
        return ts;
    }
    virtual int Pull(const std::vector<int>& keys, std::vector<float>* vals, const Callback& cb = nullptr) override {
        int ts = kvworker_->Pull(model_id_, keys, vals, cb);
        unfinished_timestamps_.insert(ts);
        return ts;
    }
    
    void Wait(int timestamp) {
        kvworker_->Wait(model_id_, timestamp);
    }
    void Erase(int timestamp) {
        unfinished_timestamps_.erase(timestamp);
    }

   private:
    int model_id_;
    int local_id_;
    kvstore::KVWorker* kvworker_ = nullptr;

    std::unordered_set<int> unfinished_timestamps_;  // keep track of the unfinished timestamps
};

}  // namespace ps
}  // namespace ml
