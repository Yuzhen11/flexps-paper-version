#pragma once

#include <vector>
#include <list>
#include <sstream>

#include "boost/iterator/indirect_iterator.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/thread/locks.hpp"
#include "boost/thread/condition_variable.hpp"
#include "core/constants.hpp"
#include "kvstore/kvstore.hpp"
#include "ml/model/chunk_based_model.hpp"

namespace ml {
namespace model {

template<typename Val>
class PushBuffer {
   public:
    PushBuffer(int model_id, int num_params):
        model_id_(model_id), num_params_(num_params),
        num_chunks_(kvstore::RangeManager::Get().GetChunkNum(model_id)),
        params_(num_chunks_), mtx_(num_chunks_) {
    }
    void PushChunks(const std::vector<size_t>& chunk_keys, const std::vector<std::vector<Val>*>& chunk_vals) {
        for (size_t i = 0; i < chunk_keys.size(); ++ i) {
            size_t chunk_id = chunk_keys[i];
            boost::lock_guard<boost::mutex> chunk_lock(mtx_[chunk_id]);
            if (params_[chunk_id].empty())
                params_[chunk_id].resize(chunk_vals[i]->size());
            for (size_t j = 0; j < chunk_vals[i]->size(); ++ j) {
                params_[chunk_id][j] += (*(chunk_vals[i]))[j];
            }
        }
    }
    void Flush(int local_id) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
        int ts;
        {
            boost::lock_guard<boost::mutex> chunk_lock(big_mutex_);

            std::vector<size_t> chunk_keys;
            std::vector<std::vector<Val>> chunk_vals;
            for (size_t i = 0; i < params_.size(); ++ i) {  // iterate over the params_
                boost::lock_guard<boost::mutex> chunk_lock(mtx_[i]);
                if (!params_[i].empty()) {
                    chunk_keys.push_back(i);
                    chunk_vals.push_back(std::move(params_[i]));
                    assert(params_[i].empty());
                }
            }
            std::vector<std::vector<Val>*> chunk_vals_ptr(chunk_vals.size());
            for (int i = 0; i < chunk_vals_ptr.size(); ++ i) {
                chunk_vals_ptr[i] = &chunk_vals[i];
            }

            ts = kvworker->PushChunks(model_id_, chunk_keys, chunk_vals_ptr);
        }
        // kvworker->Wait(model_id_, ts);
    }
   private:
    int model_id_;
    int num_params_;
    int num_chunks_;
    std::vector<std::vector<Val>> params_;
    std::vector<boost::mutex> mtx_;
    boost::mutex big_mutex_;
};

template<typename Val>
class ChunkBasedPSModel {
   public:
    ChunkBasedPSModel(int model_id, int num_params):
        model_id_(model_id), num_params_(num_params),
        num_chunks_(kvstore::RangeManager::Get().GetChunkNum(model_id)),
        params_(num_chunks_), fetch_mgr_(num_chunks_), chunk_clocks_(num_chunks_, -1), mtx_(num_chunks_) {}

    int PullWithMinClock(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals, int local_id, int min_clock) {
        // Prepare the keys
        Prepare(keys, local_id, min_clock);

        int clock;
        vals->resize(keys.size());

        auto& range_manager = kvstore::RangeManager::Get();
        size_t current_chunk_id;
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            auto chunk_id = loc.first;
            if (i == 0 || chunk_id != current_chunk_id) {
                if (i != 0) {
                    mtx_[current_chunk_id].unlock();  // unlock the current chunk
                    if (chunk_clocks_[chunk_id] < clock) clock = chunk_clocks_[chunk_id];
                } else clock = chunk_clocks_[chunk_id];
                mtx_[chunk_id].lock();  // lock the new chunk
                current_chunk_id = chunk_id;
            }
            (*vals)[i] = params_[chunk_id][loc.second];
        }
        if (keys.size() > 0) {
            mtx_[current_chunk_id].unlock();  // unlock the last chunk
        }
        
        return clock;
    }

    void PullChunksWithMinClock(const std::vector<size_t>& chunks, std::vector<std::vector<Val>*>& chunk_ptrs, int local_id, int min_clock, std::vector<int>* chunk_clocks = nullptr) {
        // Collect chunks that are too stale
        std::vector<size_t> chunks_to_fetch;
        for (auto chunk_id : chunks) {
            if (chunk_clocks_[chunk_id] < min_clock) {
                chunks_to_fetch.push_back(chunk_id);
            }
        }

        // Pull from kvstore
        if (!chunks_to_fetch.empty()) {
            // fetch_chunk(chunks_to_fetch, local_id);
            fetch_mgr_fetch_chunk(chunks_to_fetch, local_id, min_clock);
        }

        for (int i = 0; i < chunks.size(); ++i) {
            mtx_[chunks[i]].lock();
            *chunk_ptrs[i] = params_[chunks[i]];
            if (chunk_clocks != nullptr)
                (*chunk_clocks)[chunks[i]] = chunk_clocks_[chunks[i]];
            mtx_[chunks[i]].unlock();
        }
    }

    virtual void Prepare(const std::vector<husky::constants::Key>& keys, int local_id, int min_clock) {
        if (keys.empty()) return;
        std::vector<size_t> chunks_to_fetch;

        // 1. Collect all chunks that are not fresh enough
        size_t current_chunk_id;
        auto& range_manager = kvstore::RangeManager::Get();
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            auto chunk_id = loc.first;
            if ((i == 0 || chunk_id != current_chunk_id)) {
                if (chunk_clocks_[chunk_id] < min_clock) {  // the chunk is too stale
                    chunks_to_fetch.push_back(chunk_id);
                }
            }
            current_chunk_id = chunk_id;
        }

        // 2. Pull from kvstore
        if (!chunks_to_fetch.empty()) {
            fetch_chunk(chunks_to_fetch, local_id);
        }
    }

    Val At(husky::constants::Key key) {
        auto& range_manager = kvstore::RangeManager::Get();
        const auto& loc = range_manager.GetLocation(model_id_, key);
        {
            boost::lock_guard<boost::mutex> chunk_lock(mtx_[loc.first]);
            return params_[loc.first][loc.second];
        }
    }

   protected:
    /*
     * Use fetch_mgr to fetch chunk
     * Avoid repetitive fetch
     */
    void fetch_mgr_fetch_chunk(std::vector<size_t>& chunks_to_fetch, int local_id, int min_clock) {
        // Repeatedly fetch and wait
        while (!chunks_to_fetch.empty()) {
            // 1. Identify chunks_to_fetch_real and chunks_to_wait 
            std::vector<size_t> chunks_to_fetch_real;
            std::vector<size_t> chunks_to_wait;
            {
                boost::mutex::scoped_lock lock(fetch_mgr_mtx_);
                fetch_mgr_cv_.wait(lock, [this, min_clock, &chunks_to_fetch, &chunks_to_wait, &chunks_to_fetch_real]() {
                    chunks_to_fetch_real.clear();
                    chunks_to_wait.clear();
                    for (auto id : chunks_to_fetch) {
                        if (chunk_clocks_[id] < min_clock) {  // may need to lock chunk_clocks_[id]
                            // the cache is not new enough, either fetch or wait
                            auto& clock_list = fetch_mgr_[id];
                            if (!clock_list.empty() && min_clock == clock_list.front().first) {
                                if (clock_list.front().second == 1) {  // someone is working on it
                                    chunks_to_wait.push_back(id);
                                } else {  // I can work on it
                                    clock_list.front().second = 1;
                                    chunks_to_fetch_real.push_back(id);
                                }
                            } else if (!clock_list.empty() && min_clock > clock_list.front().first) {
                                // insert into the list
                                auto it = clock_list.begin();
                                while (it != clock_list.end()) {
                                    if (it->first == min_clock) {
                                        break;
                                    }
                                    if (it->first > min_clock) {
                                        clock_list.insert(it, {min_clock, 0});
                                        break;
                                    }
                                    ++ it;
                                }
                                if (min_clock > clock_list.back().first)
                                    clock_list.push_back({min_clock, 0});
                                // wait for it
                                chunks_to_wait.push_back(id);
                            } else {
                                // insert into the list
                                clock_list.push_front({min_clock, 1});
                                // fetch it
                                chunks_to_fetch_real.push_back(id);
                            }
                        }
                    }
                    if (chunks_to_fetch_real.empty() && chunks_to_wait.empty()) {  // all are clear
                        return true;
                    } else if (!chunks_to_fetch_real.empty()) {  // something to fetch
                        return true;
                    } else {
                        chunks_to_fetch.swap(chunks_to_wait);
                        return false;
                    }
                });
            }

            if (chunks_to_fetch_real.empty() && chunks_to_wait.empty())
                break;
            // 2. Fetch chunks and wait
            int fetch_min_clock = fetch_chunk(chunks_to_fetch_real, local_id);
            if (fetch_min_clock < min_clock) {
                husky::LOG_I << "fetch_min_clock vs min_clock: " << fetch_min_clock << " " << min_clock;
                assert(false);
            }
            // 3. Update fetch_mgr_
            {
                boost::mutex::scoped_lock lock(fetch_mgr_mtx_);
                // Erase from fetch_mgr
                for (auto id : chunks_to_fetch_real) {
                    auto& clock_list = fetch_mgr_[id];
                    auto it = clock_list.begin();
                    while (it != clock_list.end()) {
                        if (it->first > fetch_min_clock)
                            break;
                        it = clock_list.erase(it);
                    }
                }
                // Notify all
                fetch_mgr_cv_.notify_all();
            }
            // 4. Set chunks_to_fetch in next round
            chunks_to_fetch.swap(chunks_to_wait);
        }
    }
    int fetch_chunk(const std::vector<size_t>& chunks, int local_id) {
        int clock;
        // 1. get kvworker
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);

        // 2. pull chunks
        std::vector<std::vector<Val>*> chunk_ptrs;
        chunk_ptrs.reserve(chunks.size());
        std::vector<std::vector<Val>> tmp_chunks(chunks.size());
        for (int i = 0; i < chunks.size(); ++i) { chunk_ptrs.push_back(&tmp_chunks[i]); }
        auto ts = kvworker->PullChunksWithMinClock(this->model_id_, chunks, chunk_ptrs, &clock);
        kvworker->Wait(this->model_id_, ts);

        // 3. update chunk clocks
        for (int i = 0; i < chunks.size(); ++i) {
            auto chunk_id = chunks[i];
            boost::lock_guard<boost::mutex> chunk_lock(mtx_[chunk_id]);
            if (chunk_clocks_[chunk_id] < clock) {
                chunk_clocks_[chunk_id] = clock;
                params_[chunk_id] = std::move(tmp_chunks[i]);
            }
        }
        return clock;
    }

    void print_debug_info(const std::vector<size_t>& chunks_to_fetch, 
            const std::vector<size_t>& chunks_to_fetch_real, 
            const std::vector<size_t>& chunks_to_wait,
            int min_clock) {
        {
            boost::mutex::scoped_lock lock(fetch_mgr_mtx_);
            std::stringstream ss;
            ss << "chunks_to_fetch: ";
            for (auto id : chunks_to_fetch)
                ss << id << " ";
            ss << "\nchunks_to_fetch_real: ";
            for (auto id : chunks_to_fetch_real)
                ss << id << " ";
            ss << "\nchunks_to_wait: ";
            for (auto id : chunks_to_wait)
                ss << id << " ";
            ss << "\nmin_clock: " << min_clock;
            ss << "\nclock_list: ";
            for (auto& clock_list : fetch_mgr_) {
                for (auto it = clock_list.begin(); it != clock_list.end(); ++ it) {
                    ss << it->first << " ";
                }
                ss << "\n";
            }
            ss << "chunks_clocks: ";
            for (int i = 0; i < chunk_clocks_.size(); ++ i) {
                mtx_[i].lock();
                ss << chunk_clocks_[i] << " ";
                mtx_[i].unlock();
            }
            husky::LOG_I << ss.str();
        }
    }

    int model_id_;
    int num_params_;
    int num_chunks_;
    std::vector<std::vector<Val>> params_;
    std::vector<int> chunk_clocks_;
    std::vector<boost::mutex> mtx_;

    // for fetch_mgr
    boost::mutex fetch_mgr_mtx_;
    boost::condition_variable fetch_mgr_cv_;
    std::vector<std::list<std::pair<int, int>>> fetch_mgr_;
};

template<typename Val>
class ChunkBasedModelWithClocks : public ChunkBasedModel<Val> {
   public:
   using ChunkBasedModel<Val>::num_chunks_;
   using ChunkBasedModel<Val>::model_id_;
   using ChunkBasedModel<Val>::params_;
   using ChunkBasedModel<Val>::is_cached_;

    ChunkBasedModelWithClocks(int model_id, int num_params) :
        ChunkBasedModel<Val>(model_id, num_params),
        chunk_clocks_(num_chunks_, -1),
        range_manager_(&kvstore::RangeManager::Get()) {}

    inline void SetStaleness(int staleness) { staleness_ = staleness; }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) override {
        // 1. Update local model
        ChunkBasedModel<Val>::Push(keys, vals);

        // 2. Inc clock
        ++clock_;
    }

    virtual void PushChunks(const std::vector<size_t>& chunk_keys, const std::vector<std::vector<Val>*>& chunk_vals) override {
        // 1. Update local model
        ChunkBasedModel<Val>::PushChunks(chunk_keys, chunk_vals);

        // 2. Inc clock
        ++clock_;
    }

    virtual void PrepareChunks(const std::vector<size_t>& chunk_keys, int local_id) override {
        if (chunk_keys.empty()) return;
        std::vector<size_t> chunks_to_fetch;
        int stalest = clock_ - staleness_;
        for (auto chunk_key : chunk_keys) {
            assert(chunk_key < is_cached_.size());
            // Additional condition to fetch: staleness
            if (is_cached_[chunk_key] == false || chunk_clocks_[chunk_key] < stalest) {
                chunks_to_fetch.push_back(chunk_key);
                is_cached_[chunk_key] = true;
            }
        }
        if (chunks_to_fetch.empty()) return;
        fetch_chunk(chunks_to_fetch, local_id);
    }


    Val At(const husky::constants::Key& idx) {
        auto loc = range_manager_->GetLocation(model_id_, idx);
        return params_[loc.first][loc.second];
    }

    void Inc(const husky::constants::Key& idx, const Val& val) {
        auto loc = range_manager_->GetLocation(model_id_, idx);
        params_[loc.first][loc.second] += val;
    }

   private:
    int fetch_chunk(const std::vector<size_t>& chunks, int local_id) override {
        // 1. Get kvworker
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);

        // 2. Pull Chunks
        std::vector<std::vector<Val>*> chunk_ptrs(chunks.size());
        for (int i = 0; i < chunks.size(); ++i) { chunk_ptrs[i] = &params_[chunks[i]]; }
        int clock;
        auto ts = kvworker->PullChunksWithMinClock(model_id_, chunks, chunk_ptrs, &clock);
        kvworker->Wait(model_id_, ts);

        // 3. Update chunk clocks
        for (auto chunk_id : chunks) {
            chunk_clocks_[chunk_id] = clock;
        }
        
        return clock;
    }

    int clock_ = 0;
    int staleness_ = 1;
    std::vector<int> chunk_clocks_;
    kvstore::RangeManager* range_manager_ = nullptr;
};

/*
 */
template<typename Val>
class PSBspModel {
   public:
    PSBspModel(int model_id, int num_params, int num_local_threads) :
        model_id_(model_id), num_params_(num_params), num_local_threads_(num_local_threads),
        params_(num_params), process_cache_keys_(num_params) {}

    void Push(const std::vector<size_t>& keys, const std::vector<Val>& vals, int local_id, bool is_leader) {
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
            ts = kvworker->Push(model_id_, push_keys, push_vals);
            std::fill(params_.begin(), params_.end(), 0);
        } else {
            std::vector<Val> tmp;
            ts = kvworker->Push(model_id_, {}, tmp);
        }
        kvworker->Wait(model_id_, ts);
    }
    void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals, int local_id) {
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
                kvworker->Wait(model_id_, kvworker->Pull(model_id_, pull_keys_, &pull_vals_));
                pull_cv_.notify_all();
            } else {
                std::vector<Val> tmp;
                kvworker->Pull(model_id_, {}, &tmp);
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

}  // namespace model
}  // namespace ml
