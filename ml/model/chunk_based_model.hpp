#pragma once

#include <cassert>
#include <condition_variable>
#include <mutex>
#include <set>
#include <vector>

#include "core/constants.hpp"
#include "ml/model/model.hpp"
#include "kvstore/kvstore.hpp"

namespace ml {
namespace model {

class ChunkBasedModel : public Model {
   public:
    ChunkBasedModel(int model_id, int num_params):
        Model(model_id, num_params),
        params_(kvstore::RangeManager::Get().GetChunkNum(model_id)),
        lock_table_(kvstore::RangeManager::Get().GetChunkNum(model_id), 0),
        is_cached_(kvstore::RangeManager::Get().GetChunkNum(model_id), false) {}

    void Load(int local_id) override {}
    void Dump(int local_id) override {
        // TODO: LRU/LFU/else? replacement threshold?
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        // chunks must be pulled before push
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = kvstore::RangeManager::Get().GetLocation(model_id_, keys[i]);
            assert(is_cached_[loc.first]);
            params_[loc.first][loc.second] += vals[i];
        }
    }

    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals, int local_id) override {
        Prepare(keys, local_id);
        vals->resize(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = kvstore::RangeManager::Get().GetLocation(model_id_, keys[i]);
            (*vals)[i] = params_[loc.first][loc.second];
        }
    }
    
    virtual void Prepare(const std::vector<husky::constants::Key>& keys, int local_id) {
        // get all missing chunks
        std::set<size_t> chunks_to_fetch;
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = kvstore::RangeManager::Get().GetLocation(model_id_, keys[i]);
            if (is_cached_[loc.first] == false) {
                chunks_to_fetch.insert(loc.first);
            }
        }
        
        // if there is any missing chunk, lock and pull with kvworker
        if (chunks_to_fetch.size() > 0) {
            {
                std::unique_lock<std::mutex> table_lock(lock_table_mtx_);

                for (auto chunk_id : chunks_to_fetch) {
                    if (is_cached_[chunk_id]) chunks_to_fetch.erase(chunk_id);
                }
                if (chunks_to_fetch.size() == 0) return;  // no missing chunks now

                bool w_access = try_lock_write(chunks_to_fetch);
                while (!w_access) {
                    cv_.wait(table_lock);
                    // remove all cached chunks
                    for (auto chunk_id : chunks_to_fetch) {
                        if (is_cached_[chunk_id]) chunks_to_fetch.erase(chunk_id);
                    }
                    if (chunks_to_fetch.size() == 0) return;  // no missing chunks now
                    w_access = try_lock_write(chunks_to_fetch);
                }

            }

            auto ts = fetch_chunk({chunks_to_fetch.begin(), chunks_to_fetch.end()}, local_id);
            husky::LOG_I << "fetch " << chunks_to_fetch.size() << " chunks";
            wait(ts, local_id);

            {
                std::lock_guard<std::mutex> table_lock(lock_table_mtx_);
                // update is_cached_
                for (auto chunk_id : chunks_to_fetch) is_cached_[chunk_id] = true;
                unlock_write(chunks_to_fetch);
            }
        }
    }

   protected:
    bool try_lock_write(const std::set<size_t>& w_chunks) {  // called when lock_table_ is locked
        for (auto w : w_chunks) {
            if (lock_table_[w] != 0) return false;
        }
        // lock chunks
        for (auto w : w_chunks) { lock_table_[w] = -1; }
        return true;
    }

    void unlock_write(const std::set<size_t>& chunks) {
        for (auto chunk : chunks) {
            lock_table_[chunk] = 0;
        }
        cv_.notify_all();
    }

    int fetch_chunk(const std::vector<size_t>& chunks, int local_id) {
        assert(chunks.size() > 0);
        // 1. get kvworker
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
        // 2. pull chunks and return ts
        std::vector<std::vector<float>*> chunk_ptrs;
        chunk_ptrs.reserve(chunks.size());
        for (auto chunk_id : chunks) {
            chunk_ptrs.push_back(&(params_[chunk_id]));
        }
        return kvworker->PullChunks(this->model_id_, chunks, chunk_ptrs, false);
    }

    void wait(int ts, int local_id) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
        kvworker->Wait(this->model_id_, ts);
    }

    std::vector<std::vector<float>> params_;  // params in chunks
    std::vector<bool> is_cached_;  // indicate whether chunk has been pulled from kvstore

    std::mutex lock_table_mtx_;  // for the lock table
    std::vector<int> lock_table_;  // 0 for no lock, positive for read lock, -1 for write lock
    std::condition_variable cv_;
};


// TODO: handle writer starvation (prioritize write or read?)
class ChunkBasedLockModel : public ChunkBasedModel {
   public:
    ChunkBasedLockModel(int model_id, int num_params):
        ChunkBasedModel(model_id, num_params) {} 

    /*
     * update locally by chunks
     * 1. acquire locks atomically
     *     1.1 lock lock_table
     *     1.2 wait if any lock cannot be acquired
     *     1.3 else lock all chunks requested and unlock lock_table
     * 2. do the updates
     *     2.1 find the chunk and index
     *     2.2 update
     * 3. unlock chunks
     *     3.1 lock lock_table
     *     3.2 unlock chunks
     *     3.3 unlock lock_table and notify all
     */
    void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        // get chunk indices 
        std::set<size_t> chunks;
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = kvstore::RangeManager::Get().GetLocation(model_id_, keys[i]);
            chunks.insert(loc.first);
        }

        {
            std::unique_lock<std::mutex> table_lock(lock_table_mtx_);
            bool available = try_lock_write(chunks);
            while (!available) {
                cv_.wait(table_lock);
                available = try_lock_write(chunks);
            }
        }

        ChunkBasedModel::Push(keys, vals);

        {
            std::lock_guard<std::mutex> table_lock(lock_table_mtx_);
            unlock_write(chunks);
        }
    }

    /* get local cache and fetch on miss
     * 1. acquire locks atomically
     *     1.1 lock lock_table
     *     1.2 wait if any lock cannot be acquired
     *     1.3 else lock all chunks requested and unlock lock_table
     * 2. get chunks
     *     2.1 find the chunk and index
     *     2.2 copy
     * 3. unlock chunks
     *     3.1 lock lock_table
     *     3.2 unlock chunks
     *     3.3 unlock lock_table
     */
    void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals, int local_id) override {
        Prepare(keys, local_id);

        std::set<size_t> chunks;
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = kvstore::RangeManager::Get().GetLocation(model_id_, keys[i]);
            chunks.insert(loc.first);
        }

        {
            std::unique_lock<std::mutex> table_lock(lock_table_mtx_);
            bool r_access = try_lock_read(chunks);
            while (!r_access) {
                cv_.wait(table_lock);
                r_access = try_lock_read(chunks);
            }
        }

        vals->resize(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = kvstore::RangeManager::Get().GetLocation(model_id_, keys[i]);
            (*vals)[i] = params_[loc.first][loc.second];
        }

        {
            std::unique_lock<std::mutex> table_lock(lock_table_mtx_);
            unlock_read(chunks);
        }
    }

   protected:
    bool try_lock_read(const std::set<size_t>& chunks) {
        for (auto chunk_id : chunks) {
            if (lock_table_[chunk_id] == -1) return false;
        }

        // lock chunks
        for (auto chunk_id : chunks) { lock_table_[chunk_id] += 1; }
        return true;
    }

    void unlock_read(const std::set<size_t>& chunks) {
        bool notify = false;
        for (auto chunk_id : chunks) {
            assert(lock_table_[chunk_id] > 0);
            lock_table_[chunk_id] -= 1;
            if (lock_table_[chunk_id] == 0) notify = true;
        }
        if (notify) cv_.notify_all();
    }
};

}  // namespace model
}  // namespace ml
