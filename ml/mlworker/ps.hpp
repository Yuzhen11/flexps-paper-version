#pragma once

#include "kvstore/kvstore.hpp"
#include "ml/model/chunk_based_ps_model.hpp"
#include "ml/model/model.hpp"
#include "ml/shared/shared_state.hpp"

namespace ml {
namespace mlworker {

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
template<typename Val>
class PSWorker : public mlworker::GenericMLWorker<Val> {
   public:
    PSWorker() = delete;
    PSWorker(const PSWorker&) = delete;
    PSWorker& operator=(const PSWorker&) = delete;
    PSWorker(PSWorker&&) = delete;
    PSWorker& operator=(PSWorker&&) = delete;

    PSWorker(const husky::Info& info) 
        : model_id_(static_cast<husky::MLTask*>(info.get_task())->get_kvstore()) {
        // set kvworker
        int local_id = info.get_local_id();
        kvworker_ = kvstore::KVStore::Get().get_kvworker(local_id);
        auto& hint = info.get_task()->get_hint();
        if (hint.find(husky::constants::kConsistency) != hint.end()
                && (hint.at(husky::constants::kConsistency) == husky::constants::kSSP 
                    || hint.at(husky::constants::kConsistency) == husky::constants::kBSP)) {
            kvworker_->Wait(model_id_, kvworker_->InitForConsistencyControl(model_id_));
        }
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) override {
        assert(push_count_ + 1 == pull_count_);
        push_count_ += 1;
        ts_ = kvworker_->Push(model_id_, keys, vals, true, true);
        kvworker_->Wait(model_id_, ts_);
    }
    
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals) override {
        assert(push_count_ == pull_count_);
        pull_count_ += 1;
        if (ts_ != -1)
            kvworker_->Wait(model_id_, ts_);  // Wait for last Push, TODO: Will this cause anything wrong when changing epochs?
        ts_ = kvworker_->Pull(model_id_, keys, vals, true, true);
        kvworker_->Wait(model_id_, ts_);  // Wait for this Pull
    }

    virtual void PushChunks(const std::vector<size_t>& chunk_keys, const std::vector<std::vector<Val>*>& chunk_vals) override {
        assert(++push_count_ == pull_count_);
        assert(chunk_keys.size() == chunk_vals.size());
        ts_ = kvworker_->PushChunks(model_id_, chunk_keys, chunk_vals, true, true);
        kvworker_->Wait(model_id_, ts_);
    }

    virtual void PullChunks(const std::vector<size_t>& chunk_keys, std::vector<std::vector<Val>*>& chunk_vals) override {
        assert(push_count_ == pull_count_);
        pull_count_ += 1;
        assert(chunk_keys.size() == chunk_vals.size());
        ts_ = kvworker_->PullChunks(model_id_, chunk_keys, chunk_vals, true, true);
        kvworker_->Wait(model_id_, ts_);
    }

    // For v2
    virtual void Prepare_v2(const std::vector<husky::constants::Key>& keys) override {
        keys_ = const_cast<std::vector<husky::constants::Key>*>(&keys);
        Pull(keys, &vals_);
        delta_.clear();
        delta_.resize(keys.size());
    }
    virtual Val Get_v2(husky::constants::Key idx) override { return vals_[idx]; }
    virtual void Update_v2(husky::constants::Key idx, Val val) override {
        delta_[idx] += val;
        vals_[idx] += val;
    }
    virtual void Update_v2(const std::vector<Val>& vals) override {
        assert(vals.size() == vals_.size());
        for (size_t i = 0; i < vals.size(); ++i) {
            vals_[i] += vals[i];
            delta_[i] += vals[i];
        }
    }
    virtual void Clock_v2() override { Push(*keys_, delta_); }

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
    std::vector<Val> vals_;
    std::vector<Val> delta_;
};

/*
 * PSMapNoneWorker
 * ThreadCache: unordered_map, one timestamp
 * ProcessCache: No
 */
template<typename Val>
class PSMapNoneWorker : public mlworker::GenericMLWorker<Val> {
   public:
    PSMapNoneWorker() = delete;
    PSMapNoneWorker(const PSMapNoneWorker&) = delete;
    PSMapNoneWorker& operator=(const PSMapNoneWorker&) = delete;
    PSMapNoneWorker(PSMapNoneWorker&&) = delete;
    PSMapNoneWorker& operator=(PSMapNoneWorker&&) = delete;

    PSMapNoneWorker(const husky::Info& info)
        : model_id_(static_cast<husky::MLTask*>(info.get_task())->get_kvstore()) {
        // set staleness
        staleness_ = stoi(info.get_task()->get_hint().at(husky::constants::kStaleness));
        // set kvworker
        int local_id = info.get_local_id();
        kvworker_ = kvstore::KVStore::Get().get_kvworker(local_id);
        kvworker_->Wait(model_id_, kvworker_->InitForConsistencyControl(model_id_));
    }
    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) override {
        assert(push_count_ + 1 == pull_count_);
        push_count_ += 1;
        ts_ = kvworker_->Push(model_id_, keys, vals);
        // update local cache but not cache timestamp
        for (int i = 0; i < keys.size(); i++) {
            cached_kv_[keys[i]] += vals[i];
        }
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals) override {
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
        if (pull_count_ - cache_ts_ > staleness_ || cached_kv_.size() == 0) {
            uncached_keys = keys;
        } else {
            for (auto& key : keys) {
                if (cached_kv_.find(key) == cached_kv_.end()) {
                    uncached_keys.push_back(key);
                }
            }
        }

        if (uncached_keys.size() > 0) {
            ts_ = kvworker_->Pull(model_id_, uncached_keys, vals);
            kvworker_->Wait(model_id_, ts_);

            // Clear cache and update cache_ts_
            if (uncached_keys.size() == keys.size()) {
                cached_kv_.clear();
                cache_ts_ = pull_count_;
            }
            for (int i = 0; i < uncached_keys.size(); i++) {
                cached_kv_[uncached_keys[i]] = (*vals)[i];
            }
        }

        // update all vals using cache
        vals->resize(keys.size());
        for (int i = 0; i < keys.size(); ++i) {
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
    virtual Val Get_v2(husky::constants::Key idx) override { return vals_[idx]; }
    virtual void Update_v2(husky::constants::Key idx, Val val) override {
        delta_[idx] += val;
        vals_[idx] += val;
    }
    virtual void Update_v2(const std::vector<Val>& vals) override {
        assert(vals.size() == vals_.size());
        for (int i = 0; i < vals.size(); ++i) {
            vals_[i] += vals[i];
            delta_[i] += vals[i];
        }
    }
    virtual void Clock_v2() override { Push(*keys_, delta_); }

   private:
    int model_id_;
    int staleness_ = -1;
    int cache_ts_;
    kvstore::KVWorker* kvworker_ = nullptr;
    std::unordered_map<husky::constants::Key, Val> cached_kv_;  // timestamp, key_val dictionary

    // Just to restrict the usage of the Push/Pull APIs,
    // The correct usage should be Pull, Push, Pull, Push...
    int push_count_ = 0;
    int pull_count_ = 0;
    int ts_ = -1;

    // For v2
    // Pointer to keys
    std::vector<husky::constants::Key>* keys_;
    std::vector<Val> vals_;
    std::vector<Val> delta_;
};

/*
 * PSChunkNoneWorker
 * ThreadCache: Chunk-based
 * ProcessCache: No
 */
template<typename Val>
class PSChunkNoneWorker : public mlworker::GenericMLWorker<Val> {
   public:
    PSChunkNoneWorker() = delete;
    PSChunkNoneWorker(const PSChunkNoneWorker&) = delete;
    PSChunkNoneWorker operator=(const PSChunkNoneWorker&) = delete;
    PSChunkNoneWorker(PSChunkNoneWorker&&) = delete;
    PSChunkNoneWorker operator=(PSChunkNoneWorker&&) = delete;

    PSChunkNoneWorker(const husky::Info& info) :
        model_id_(static_cast<husky::MLTask*>(info.get_task())->get_kvstore()),
        model_(model_id_, static_cast<husky::MLTask*>(info.get_task())->get_dimensions()),
        local_id_(info.get_local_id()) {
            // Configure model
            model_.SetStaleness(stoi(info.get_task()->get_hint().at(husky::constants::kStaleness)));
            // Set kvworker
            kvworker_ = kvstore::KVStore::Get().get_kvworker(local_id_);
            kvworker_->Wait(model_id_, kvworker_->InitForConsistencyControl(model_id_));
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) override {
        assert(++push_count_ == pull_count_);
        // 1. Push updates to kvstore
        ts_ = kvworker_->Push(model_id_, keys, vals);

        // 2. Update local model
        model_.Push(keys, vals);
    }

    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals) override {
        assert(push_count_ == pull_count_);
        ++pull_count_;

        if(ts_ != -1) kvworker_->Wait(model_id_, ts_);

        model_.Pull(keys, vals, local_id_);
    }
    virtual void PushChunks(const std::vector<size_t>& chunk_keys, const std::vector<std::vector<Val>*>& chunk_vals) override {
        assert(++push_count_ == pull_count_);
        assert(chunk_keys.size() == chunk_vals.size());
        ts_ = kvworker_->PushChunks(model_id_, chunk_keys, chunk_vals, true, true);
        model_.PushChunks(chunk_keys, chunk_vals);
    }
    virtual void PullChunks(const std::vector<size_t>& chunk_keys, std::vector<std::vector<Val>*>& chunk_vals) override {
        assert(push_count_ == pull_count_);
        pull_count_ += 1;
        if(ts_ != -1) kvworker_->Wait(model_id_, ts_);
        assert(chunk_keys.size() == chunk_vals.size());
        model_.PullChunks(chunk_keys, chunk_vals, local_id_);
    }

    virtual void Prepare_v2(const std::vector<husky::constants::Key>& keys) override {
        ++pull_count_;
        keys_ = const_cast<std::vector<husky::constants::Key>*>(&keys);
        model_.Prepare(keys, local_id_);
        delta_.clear();
        delta_.resize(keys.size());
    }

    virtual Val Get_v2(husky::constants::Key idx) override {
        return model_.At((*keys_)[idx]);
    }
    virtual void Update_v2(husky::constants::Key idx, Val val) override {
        delta_[idx] += val;
        model_.Inc((*keys_)[idx], val);
    }
    virtual void Update_v2(const std::vector<Val>& vals) override {
        for (int i = 0; i < vals.size(); ++i) {
            model_.Inc((*keys_)[i], vals[i]);
            delta_[i] += vals[i];
        }
    }
    virtual void Clock_v2() override { Push(*keys_, delta_); }

   private:
    int model_id_;
    int staleness_ = 1;
    kvstore::KVWorker* kvworker_ = nullptr;
    int local_id_;

    int push_count_ = 0;
    int pull_count_ = 0;
    int ts_ = -1;

    model::ChunkBasedModelWithClocks<Val> model_;
    // For v2
    std::vector<husky::constants::Key>* keys_;
    std::vector<Val> delta_;
};

/*
 * PSNoneChunkWorker
 * ThreadCache: none
 * ProcessCache: chunk-based
 */
template<typename Val>
class PSNoneChunkWorker : public mlworker::GenericMLWorker<Val> {
    struct PSState {
        model::ChunkBasedPSModel<Val>* p_model_;
    };

   public:
    PSNoneChunkWorker() = delete;
    PSNoneChunkWorker(const PSNoneChunkWorker&) = delete;
    PSNoneChunkWorker(PSNoneChunkWorker&&) = delete;
    PSNoneChunkWorker& operator=(const PSNoneChunkWorker&) = delete;
    PSNoneChunkWorker& operator=(PSNoneChunkWorker&&) = delete;

    PSNoneChunkWorker(const husky::Info& info, zmq::context_t& context) :
        shared_state_(info.get_task_id(), info.is_leader(), info.get_num_local_workers(), context),
        info_(info),
        model_id_(static_cast<husky::MLTask*>(info.get_task())->get_kvstore()) {
         size_t num_params = static_cast<husky::MLTask*>(info_.get_task())->get_dimensions();
        if (info.is_leader()) {
            PSState* state = new PSState;
            state->p_model_ = new model::ChunkBasedPSModel<Val>(model_id_, num_params);
            // 1. Init
            shared_state_.Init(state);
        }
        // 2. Sync
        shared_state_.SyncState();
        staleness_ = stoi(info.get_task()->get_hint().at(husky::constants::kStaleness));
        // set local id and kvworker
        local_id_ = info.get_local_id();
        kvworker_ = kvstore::KVStore::Get().get_kvworker(local_id_);
        kvworker_->Wait(model_id_, kvworker_->InitForConsistencyControl(model_id_));
    }

    ~PSNoneChunkWorker() {
        shared_state_.Barrier();
        if (info_.get_local_tids().at(0) == info_.get_global_id()) {
            delete shared_state_.Get()->p_model_;
            delete shared_state_.Get();
        }
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) override {
        assert(++push_count_ == pull_count_);
        // Push updates to PS
        kvworker_->Push(model_id_, keys, vals);
    }

    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals) override {
        assert(pull_count_ == push_count_);
        ++pull_count_;
        
        int stale = std::max(pull_count_ - staleness_ - 1, 0);
        shared_state_.Get()->p_model_->PullWithMinClock(keys, vals, local_id_, stale);
    }

    virtual void PushChunks(const std::vector<size_t>& chunk_keys, const std::vector<std::vector<Val>*>& chunk_vals) override {
        assert(++push_count_ == pull_count_);
        kvworker_->PushChunks(model_id_, chunk_keys, chunk_vals);
    }

    virtual void PullChunks(const std::vector<size_t>& chunk_keys, std::vector<std::vector<Val>*>& chunk_vals) override {
        assert(pull_count_ == push_count_);
        ++pull_count_;
        
        int stale = std::max(pull_count_ - staleness_ - 1, 0);
        shared_state_.Get()->p_model_->PullChunksWithMinClock(chunk_keys, chunk_vals, local_id_, stale, nullptr);
    }

    // v2: no read-your-writes guarantee
    virtual void Prepare_v2(const std::vector<husky::constants::Key>& keys) override {
        ++pull_count_;
        keys_ = const_cast<std::vector<husky::constants::Key>*>(&keys);
        int stale = std::max(pull_count_ - staleness_ - 1, 0);
        shared_state_.Get()->p_model_->Prepare(keys, local_id_, stale);
        delta_.clear();
        delta_.resize(keys.size());
    }

    virtual Val Get_v2(husky::constants::Key idx) override {
        return shared_state_.Get()->p_model_->At((*keys_)[idx]);
    }

    virtual void Update_v2(husky::constants::Key idx, Val val) override {
        delta_[idx] += val;
    }

    virtual void Update_v2(const std::vector<Val>& vals) override {
        for (int i = 0; i < vals.size(); ++i) {
            delta_[i] += vals[i];
        }
    }
    virtual void Clock_v2() override { Push(*keys_, delta_); }

   private: 
    int model_id_;
    const husky::Info& info_;
    int local_id_;
    kvstore::KVWorker* kvworker_ = nullptr;
    // Shared Model
    SharedState<PSState> shared_state_;

    // Progress
    int pull_count_ = 0;  // clock
    int push_count_ = 0;
    int staleness_ = 1;  // default is synchronouse

    // for v2
    std::vector<husky::constants::Key>* keys_;
    std::vector<Val> delta_;
};

/*
 * PSMapChunkWorker
 * ThreadCache: unordered_map, one timestamp
 * ProcessCache: Chunk-based
 */
template<typename Val>
class PSMapChunkWorker : public mlworker::GenericMLWorker<Val> {
    struct PSState {
        model::ChunkBasedPSModel<Val>* p_model_;
    };

   public:
    PSMapChunkWorker() = delete;
    PSMapChunkWorker(const PSMapChunkWorker&) = delete;
    PSMapChunkWorker& operator=(const PSMapChunkWorker&) = delete;
    PSMapChunkWorker(PSMapChunkWorker&&) = delete;
    PSMapChunkWorker& operator=(PSMapChunkWorker&&) = delete;

    PSMapChunkWorker(const husky::Info& info, zmq::context_t& context)
        : shared_state_(info.get_task_id(), info.is_leader(), info.get_num_local_workers(), context),
          info_(info),
          model_id_(static_cast<husky::MLTask*>(info.get_task())->get_kvstore()) {
        size_t num_params = static_cast<husky::MLTask*>(info_.get_task())->get_dimensions();
        if (info.is_leader()) {
            PSState* state = new PSState;
            state->p_model_ = new model::ChunkBasedPSModel<Val>(model_id_, num_params);
            // 1. Init
            shared_state_.Init(state);
        }
        // 2. Sync
        shared_state_.SyncState();
        staleness_ = stoi(info.get_task()->get_hint().at(husky::constants::kStaleness));
        // set local id and kvworker
        local_id_ = info.get_local_id();
        kvworker_ = kvstore::KVStore::Get().get_kvworker(local_id_);
        kvworker_->Wait(model_id_, kvworker_->InitForConsistencyControl(model_id_));
    }

    ~PSMapChunkWorker() {
        shared_state_.Barrier();
        if (info_.get_local_tids().at(0) == info_.get_global_id()) {
            delete shared_state_.Get()->p_model_;
            delete shared_state_.Get();
        }
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) override {
        assert(pull_count_ == push_count_ + 1);
        ++push_count_;
        // 1. Push updates to PS
        ts_ = kvworker_->Push(model_id_, keys, vals);

        // 2. Update local model: Aggregate
        for (int i = 0; i < keys.size(); i++) {
            cached_kv_[keys[i]] += vals[i];
        }
    }

    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals) override {
        assert(pull_count_ == push_count_);
        ++pull_count_;
        // TODO: is it necessary to wait?
        if (ts_ != -1) kvworker_->Wait(model_id_, ts_);  // Wait for last Push
        
        Prepare(keys);

        vals->resize(keys.size());
        for (int i = 0; i < keys.size(); i++) {
            (*vals)[i] = cached_kv_[keys[i]];
        }
    }

    virtual void Prepare_v2(const std::vector<husky::constants::Key>& keys) override {
        ++pull_count_;
        keys_ = const_cast<std::vector<husky::constants::Key>*>(&keys);
        Prepare(keys);
        delta_.clear();
        delta_.resize(keys.size());
    }

    virtual Val Get_v2(husky::constants::Key idx) override { return cached_kv_[(*keys_)[idx]]; }
    virtual void Update_v2(husky::constants::Key idx, Val val) override {
        delta_[idx] += val;
        cached_kv_[(*keys_)[idx]] += val;
    }
    virtual void Update_v2(const std::vector<Val>& vals) override {
        for (int i = 0; i < vals.size(); ++i) {
            cached_kv_[(*keys_)[i]] += vals[i];
            delta_[i] += vals[i];
        }
    }
    virtual void Clock_v2() override {
        Push(*keys_, delta_);
    }

   private: 
    void Prepare(const std::vector<husky::constants::Key>& keys) {
        std::vector<husky::constants::Key> uncached_keys;
        // 1. Check the staleness of local model
        if (pull_count_ - cache_ts_ > staleness_ || cached_kv_.empty()) {  // Thread-cache is too old or empty
            cached_kv_.clear();
            uncached_keys = keys;
        } else {
            // 2. Collect all missing keys
            for (auto& key : keys) {
                if (cached_kv_.find(key) == cached_kv_.end()) uncached_keys.push_back(key);
            }
        }

        // 3. Pull missing keys from process cache
        if (!uncached_keys.empty()) {
            std::vector<Val> tmp_vals;
            int stale = std::max(pull_count_ - staleness_ -1, 0);
            auto cache_ts = shared_state_.Get()->p_model_->PullWithMinClock(uncached_keys, &tmp_vals, local_id_, stale);
            if (keys.size() == uncached_keys.size()) {
                cache_ts_ = cache_ts;
            }
            for (int i = 0; i < uncached_keys.size(); ++i) {
                cached_kv_[uncached_keys[i]] = tmp_vals[i];
            }
        }
    }

    int model_id_;
    const husky::Info& info_;
    int local_id_;
    kvstore::KVWorker* kvworker_ = nullptr;
    // Shared Model
    SharedState<PSState> shared_state_;
    // Local Model
    std::unordered_map<husky::constants::Key, Val> cached_kv_;  // key_val dictionary
    int cache_ts_;
    int ts_ = -1;
    
    // Progress
    int pull_count_ = 0;  // clock
    int push_count_ = 0;
    int staleness_ = 1;  // default is synchronouse

    // For v2
    std::vector<husky::constants::Key>* keys_;
    std::vector<Val> delta_;
};

/*
 * PSChunkChunkWorker
 * ThreadCache: Chunk-based
 * ProcessCache: Chunk-based
 */
template<typename Val>
class PSChunkChunkWorker : public mlworker::GenericMLWorker<Val> {
    struct PSState {
        model::ChunkBasedPSModel<Val>* p_model_;
    };

   public:
    PSChunkChunkWorker() = delete;
    PSChunkChunkWorker(const PSChunkChunkWorker&) = delete;
    PSChunkChunkWorker& operator=(const PSChunkChunkWorker&) = delete;
    PSChunkChunkWorker(PSChunkChunkWorker&&) = delete;
    PSChunkChunkWorker& operator=(PSChunkChunkWorker&&) = delete;

    PSChunkChunkWorker(const husky::Info& info, zmq::context_t& context)
        : shared_state_(info.get_task_id(), info.is_leader(), info.get_num_local_workers(), context),
          info_(info),
          model_id_(static_cast<husky::MLTask*>(info.get_task())->get_kvstore()),
          chunk_clocks_(kvstore::RangeManager::Get().GetChunkNum(model_id_), -1),
          params_(kvstore::RangeManager::Get().GetChunkNum(model_id_)) {
        size_t num_params = static_cast<husky::MLTask*>(info_.get_task())->get_dimensions();
        if (info.is_leader()) {
            PSState* state = new PSState;
            state->p_model_ = new model::ChunkBasedPSModel<Val>(model_id_, num_params);
            // 1. Init
            shared_state_.Init(state);
        }
        // 2. Sync
        shared_state_.SyncState();
        staleness_ = stoi(info.get_task()->get_hint().at(husky::constants::kStaleness));

        // Set local id and kvworker
        local_id_ = info.get_local_id();
        kvworker_ = kvstore::KVStore::Get().get_kvworker(local_id_);
        kvworker_->Wait(model_id_, kvworker_->InitForConsistencyControl(model_id_));
    }

    ~PSChunkChunkWorker() {
        shared_state_.Barrier();
        if (info_.get_local_tids().at(0) == info_.get_global_id()) {
            delete shared_state_.Get()->p_model_;
            delete shared_state_.Get();
        }
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) override {
        assert(pull_count_ == push_count_ + 1);
        ++push_count_;
        // 1. Push updates to PS
        ts_ = kvworker_->Push(model_id_, keys, vals);

        // 2. Update local model: Aggregate
        auto& range_manager = kvstore::RangeManager::Get();
        for (int i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            params_[loc.first][loc.second] += vals[i];
        }
    }

    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals) override {
        assert(pull_count_ == push_count_);
        ++pull_count_;
        // TODO: is it necessary to wait?
        if (ts_ != -1) kvworker_->Wait(model_id_, ts_);  // Wait for last Push
        
        Prepare(keys);

        vals->resize(keys.size());
        auto& range_manager = kvstore::RangeManager::Get();
        for (int i = 0; i < keys.size(); i++) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            (*vals)[i] = params_[loc.first][loc.second];
        }
    }

    virtual void PushChunks(const std::vector<size_t>& chunk_keys, const std::vector<std::vector<Val>*>& chunk_vals) override {
        assert(pull_count_ == push_count_ + 1);
        ++push_count_;
        // 1. Push updates to PS
        ts_ = kvworker_->PushChunks(model_id_, chunk_keys, chunk_vals);

        // 2. Update local model: Aggregate
        for (size_t i = 0; i < chunk_keys.size(); ++ i) {
            size_t chunk_id = chunk_keys[i];
            assert(params_[chunk_id].size() == chunk_vals[i]->size());
            for (size_t j = 0; j < chunk_vals[i]->size(); ++ j) {
                params_[chunk_id][j] += (*(chunk_vals[i]))[j];
            }
        }
    }

    virtual void PullChunks(const std::vector<size_t>& chunk_keys, std::vector<std::vector<Val>*>& chunk_vals) override {
        assert(pull_count_ == push_count_);
        ++pull_count_;
        // TODO: is it necessary to wait?
        if (ts_ != -1) kvworker_->Wait(model_id_, ts_);  // Wait for last Push
        
        PrepareChunks(chunk_keys);

        for (size_t i = 0; i < chunk_keys.size(); i++) {
            size_t chunk_id = chunk_keys[i];
            chunk_vals[i]->resize(params_[chunk_id].size());
            for (size_t j = 0; j < chunk_vals[i]->size(); j++) {
                (*(chunk_vals[i]))[j] = params_[chunk_id][j];
            }
        }
    }

    virtual void Prepare_v2(const std::vector<husky::constants::Key>& keys) override {
        ++pull_count_;
        keys_ = const_cast<std::vector<husky::constants::Key>*>(&keys);
        Prepare(keys);
        delta_.clear();
        delta_.resize(keys.size());
    }

    virtual Val Get_v2(husky::constants::Key idx) override {
        auto& range_manager = kvstore::RangeManager::Get();
        auto loc = range_manager.GetLocation(model_id_, (*keys_)[idx]);
        return params_[loc.first][loc.second];
    }

    virtual void Update_v2(husky::constants::Key idx, Val val) override {
        delta_[idx] += val;
        auto& range_manager = kvstore::RangeManager::Get();
        auto loc = range_manager.GetLocation(model_id_, (*keys_)[idx]);
        params_[loc.first][loc.second] += val;
    }

    virtual void Update_v2(const std::vector<Val>& vals) override {
        auto& range_manager = kvstore::RangeManager::Get();
        for (int i = 0; i < vals.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, (*keys_)[i]);
            params_[loc.first][loc.second] += vals[i];
            delta_[i] += vals[i];
        }
    }

    virtual void Clock_v2() override {
        Push(*keys_, delta_);
    }

   private: 
    void Prepare(const std::vector<husky::constants::Key>& keys) {
        std::vector<size_t> chunk_ids;
        auto& range_manager = kvstore::RangeManager::Get();
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            if (chunk_ids.empty() || loc.first != chunk_ids.back()) {
                chunk_ids.push_back(loc.first);
            }
        }
        PrepareChunks(chunk_ids);
    }
    void PrepareChunks(const std::vector<size_t>& chunk_keys) {
        std::vector<size_t> uncached_chunks;
        int min_clock = std::max(0, pull_count_ - staleness_ - 1);

        for (auto chunk_key : chunk_keys) {
            if (chunk_clocks_[chunk_key] < min_clock) {
                uncached_chunks.push_back(chunk_key);
            }
        }

        // 3. Pull missing chunks from process cache
        if (!uncached_chunks.empty()) {
            std::vector<std::vector<Val>*> chunk_ptrs;
            chunk_ptrs.reserve(uncached_chunks.size());
            for (auto chunk_id : uncached_chunks) {
                chunk_ptrs.push_back(&params_[chunk_id]);
            }
            shared_state_.Get()->p_model_->PullChunksWithMinClock(uncached_chunks, chunk_ptrs, local_id_, min_clock, &chunk_clocks_);
        }
    }

    int model_id_;
    const husky::Info& info_;
    int local_id_;
    kvstore::KVWorker* kvworker_ = nullptr;
    // Shared Model
    SharedState<PSState> shared_state_;
    // Local Model
    std::vector<std::vector<Val>> params_;
    std::vector<int> chunk_clocks_;
    int ts_ = -1;
    
    // Progress
    int pull_count_ = 0;  // clock
    int push_count_ = 0;
    int staleness_ = 1;  // default is synchronouse

    // For v2
    std::vector<husky::constants::Key>* keys_;
    std::vector<Val> delta_;
};

}  // namespace mlworker
}  // namespace ml
