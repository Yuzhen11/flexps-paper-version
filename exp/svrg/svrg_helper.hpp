#pragma once

#include "kvstore/kvstore.hpp"
#include "ml/model/model.hpp"
#include "ml/shared/shared_state.hpp"

/*
 */
namespace ml {
namespace mlworker2 {

template<typename Val>
class PSBspModel {
   public:
    PSBspModel(int model_id, int num_params, int num_local_threads) :
        model_id_(model_id), num_params_(num_params), num_local_threads_(num_local_threads),
        params_(num_params), process_cache_keys_(num_params) {}

    void Push(const std::vector<size_t>& keys, const std::vector<Val>& vals, int local_id, bool is_leader, bool enable_cc = true) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            for (size_t i = 0; i < keys.size(); ++ i) {
                params_[keys[i]] += vals[i];
            }
        }
        {
            std::unique_lock<std::mutex> lock(push_mtx_);
            push_num_ += 1;
            if (push_num_ == num_local_threads_) {
                push_cv_.notify_all();
            } else {
                // block until push_num_ == num_local_threads_
                push_cv_.wait(lock, [this]() {
                    return push_num_ == num_local_threads_;
                });
            }
        }
        {
            std::unique_lock<std::mutex> lock(push_mtx_);
            push_num2_ += 1;
            if (push_num2_ == num_local_threads_) {
                push_num_ = 0;
                push_num2_ = 0;
            }
        }
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
        int ts;
        if (is_leader == true) {
            // leader sends out all keys
            std::vector<size_t> push_keys;
            std::vector<Val> push_vals;
            for (int i = 0; i < params_.size(); ++ i) {
                if (params_[i] != 0) {
                    push_keys.push_back(i);
                    push_vals.push_back(params_[i]);
                }
            }
            ts = kvworker->Push(model_id_, push_keys, push_vals, true, true, enable_cc);
            std::fill(params_.begin(), params_.end(), 0);
        } else {
            std::vector<Val> tmp;
            ts = kvworker->Push(model_id_, {}, tmp, true, true, enable_cc);
        }
        kvworker->Wait(model_id_, ts);
    }
    void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals, int local_id, bool enable_cc = true) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            for (auto k : keys) {
                process_cache_keys_[k] = 1;
            }
        }
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
        {
            std::unique_lock<std::mutex> lock(pull_mtx_);
            pull_num_ += 1;
            if (pull_num_ == num_local_threads_) {
                pull_keys_.clear();
                pull_vals_.clear();
                // pull !
                for (int i = 0; i < process_cache_keys_.size(); ++ i) {
                    if (process_cache_keys_[i] == 1) {
                        pull_keys_.push_back(i);
                    }
                }
                kvworker->Wait(model_id_, kvworker->Pull(model_id_, pull_keys_, &pull_vals_, true, true, enable_cc));
                pull_cv_.notify_all();
            } else {
                std::vector<Val> tmp;
                kvworker->Pull(model_id_, {}, &tmp, true, true, enable_cc);
                pull_cv_.wait(lock, [this]() {
                    return pull_num_ == num_local_threads_;
                });
            }
        }
        vals->clear();
        int i = 0;
        for (auto k : keys) {
            while (i != pull_keys_.size() && pull_keys_[i] != k) i += 1;
            assert(i != pull_keys_.size());
            vals->push_back(pull_vals_[i]);
        }
        assert(keys.size() == vals->size());
        {
            std::lock_guard<std::mutex> lock(pull_mtx_);
            pull_num2_ += 1;
            if (pull_num2_ == num_local_threads_) {  // reset
                pull_num_ = 0;
                pull_num2_ = 0;
                std::fill(process_cache_keys_.begin(), process_cache_keys_.end(), 0);
            }
        }
    }
   private:
    int num_local_threads_;
    int model_id_;
    int num_params_;

    std::mutex mtx_;
    std::vector<Val> params_;
    std::vector<int> process_cache_keys_;
    std::vector<size_t> pull_keys_;
    std::vector<Val> pull_vals_;

    int push_num_ = 0;
    int pull_num_ = 0;
    int push_num2_ = 0;
    int pull_num2_ = 0;
    std::mutex push_mtx_;
    std::mutex pull_mtx_;
    std::condition_variable push_cv_;
    std::condition_variable pull_cv_;
};

/*
 * PSBspWorker
 * Provide simple process-level cache for PSWorker in BSP mode
 * Only vector-version shared state is provided
 */
template<typename Val>
class PSBspWorker {
    struct PSState {
        PSState(int model_id, int num_params, int num_workers):
        model_(model_id, num_params, num_workers) {}
        PSBspModel<Val> model_;
    };
   public:
    PSBspWorker() = delete;
    PSBspWorker(const PSBspWorker&) = delete;
    PSBspWorker& operator=(const PSBspWorker&) = delete;
    PSBspWorker(PSBspWorker&&) = delete;
    PSBspWorker& operator=(PSBspWorker&&) = delete;

    PSBspWorker(const husky::Info& info, zmq::context_t& context, int kv_store, int num_params)
        : shared_state_(info.get_task_id() + kv_store, info.is_leader(), info.get_num_local_workers(), context),
        info_(info),
        model_id_(kv_store) {

        if (info.is_leader()) {
            PSState* state = new PSState(model_id_, num_params, info.get_num_local_workers());
            // 1. Init
            shared_state_.Init(state);
        }
        // 2. Sync
        shared_state_.SyncState();
        // set kvworker
        local_id_ = info.get_local_id();
        kvworker_ = kvstore::KVStore::Get().get_kvworker(local_id_);
        kvworker_->Wait(model_id_, kvworker_->InitForConsistencyControl(model_id_, info.get_num_workers()));
    }
    ~PSBspWorker() {
        shared_state_.Barrier();
        if (info_.get_local_tids().at(0) == info_.get_global_id()) {
            delete shared_state_.Get();
        }
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) {
        assert(push_count_ + 1 == pull_count_);
        push_count_ += 1;
        shared_state_.Get()->model_.Push(keys, vals, local_id_, info_.is_leader());
    }
    
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals) {
        assert(push_count_ == pull_count_);
        pull_count_ += 1;
        shared_state_.Get()->model_.Pull(keys, vals, local_id_); 
    }

    kvstore::KVWorker* GetKVWorker() {
        return kvworker_;
    }
   private:
    int model_id_;
    kvstore::KVWorker* kvworker_ = nullptr;
    int local_id_;
    // Shared Model
    SharedState<PSState> shared_state_;
    const husky::Info& info_;

    // Just to restrict the usage of the Push/Pull APIs,
    // The correct usage should be Pull, Push, Pull, Push...
    int push_count_ = 0;
    int pull_count_ = 0;
    int ts_ = -1;
    bool send_all_ = true;
};

}
}

