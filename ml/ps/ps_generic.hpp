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
    virtual void Prepare_v2(const std::vector<husky::constants::Key>& keys) override {
        keys_ = const_cast<std::vector<husky::constants::Key>*>(&keys);
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

class SSPWorker : public common::GenericMLWorker {
   public:
    SSPWorker() = delete;
    SSPWorker(int model_id, int local_id, int staleness): model_id_(model_id),
        kvworker_(kvstore::KVStore::Get().get_kvworker(local_id)), staleness_(staleness) {
    }
    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        assert(push_count_ + 1 == pull_count_);
        push_count_ += 1;
        ts_ = kvworker_->Push(model_id_, keys, vals, nullptr);
        // update local cache but not cache timestamp
        for (int i = 0; i < keys.size(); i++) {
            if (cached_kv_.find(keys[i]) != cached_kv_.end()) {
                cached_kv_.at(keys[i]) = vals[i];
            } else {
                cached_kv_.emplace(std::make_pair(keys[i], vals[i]));
            }
        }
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals) override {
        assert(push_count_ == pull_count_);
        pull_count_ += 1;
        if (ts_ != -1)
            kvworker_->Wait(model_id_, ts_);  // Wait for last Push

        // Cache update strategy:
        // 1. Cache is empty or too old or miss all required keys: clear cache and update cache_ts_;
        // 2. Cache is non-empty and not too old but miss some keys: only update those missed;
        // 3. Cache is non-empty and not too old and contains all keys required.

        // find uncached_keys 
        std::vector<husky::constants::Key> uncached_keys;
        if (pull_count_ - cache_ts_ < staleness_ || cached_kv_.size() == 0) {
            uncached_keys = keys;
        } else {
            for (auto& key : keys) {
                if (cached_kv_.find(key) == cached_kv_.end()) {
                    uncached_keys.push_back(key);
                }
            } 
        }

        if (uncached_keys.size() > 0) {
            ts_ = kvworker_->Pull(model_id_, uncached_keys, vals, nullptr); 
            kvworker_->Wait(model_id_, ts_);

            // Clear cache and update cache_ts_
            if (uncached_keys.size() == keys.size()) {
                if (cached_kv_.size() > 0) {
                    cached_kv_.clear();
                }
                cache_ts_ = pull_count_;
            } 
            for (int i = 0; i < uncached_keys.size(); i++) {
                cached_kv_.insert(std::make_pair(uncached_keys[i], (*vals)[uncached_keys[i]]));
            }
        } 

        // update all vals using cache
        vals->resize(keys.size());
        for (int i = 0; i < keys.size(); i++) {
            (*vals)[i] = cached_kv_[keys[i]];
        }
    }
    
    // For v2
    virtual void Prepare_v2(const std::vector<husky::constants::Key>& keys) override {
        keys_ = const_cast<std::vector<husky::constants::Key>*>(&keys);
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
        for (int i = 0; i < vals.size(); ++ i) {
            vals_[i] += vals[i];
            delta_[i] += vals[i];
        }
    }
    virtual void Clock_v2() override {
        Push(*keys_, delta_);
    }
    
   private:
    int model_id_;
    //TODO: Repaleced with user-defined staleness
    int staleness_ = -1;
    int cache_ts_;
    kvstore::KVWorker* kvworker_ = nullptr;
    std::unordered_map<husky::constants::Key, float> cached_kv_;// timestamp, key_val dictionary

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
