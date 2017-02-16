#pragma once

#include <cassert>
#include <condition_variable>
#include <mutex>
#include <set>
#include <vector>

#include "core/constants.hpp"
#include "ml/model/chunk_based_model.hpp"
#include "kvstore/kvstore.hpp"

namespace ml {
namespace model {

/*
 * ChunkBasedMTModel
 *
 * The ChunkBased Model for Push/Pull in a lock free manner for Multi-threads
 *
 * The Pull is not totally lock-free, since we need to fetch from kvstore
 *
 * Double-checked locking is used
 */
class ChunkBasedMTModel : public ChunkBasedModel {
   public:
    ChunkBasedMTModel() = delete;
    ChunkBasedMTModel(int model_id, int num_params):
        ChunkBasedModel(model_id, num_params),
        lock_table_(kvstore::RangeManager::Get().GetChunkNum(model_id), 0) {}

    void Load(int local_id) override {}
    void Dump(int local_id) override {
        // TODO: LRU/LFU/else? replacement threshold?
    }

   protected:
    /*
     * Override the Prepare function in ChunkBasedModel
     *
     * Allow concurrent Pull issues to kvstore
     * Use double-checked locking to reduce the overhead of acquiring the lock
     */
    virtual void Prepare(const std::vector<husky::constants::Key>& keys, int local_id) override {
        auto& range_manager = kvstore::RangeManager::Get();
        std::vector<size_t> chunks_to_fetch;
        // 1. Collect the uncached chunks
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            if (is_cached_[loc.first] == false) {
                chunks_to_fetch.push_back(loc.first);
            }
        }
        
        // 2. Test the locking criterion without actually acquiring the lock
        // If there is any missing chunk, lock and pull with kvworker
        if (chunks_to_fetch.size() > 0) {
            {
                // 3. Lock and check again
                std::unique_lock<std::mutex> table_lock(lock_table_mtx_);

                // Erase-remove idiom
                chunks_to_fetch.erase(std::remove_if(chunks_to_fetch.begin(), 
                                                  chunks_to_fetch.end(), 
                                                  [this](size_t chunk_id){
                                                      return is_cached_[chunk_id];
                                                  }),
                                     chunks_to_fetch.end());
                if (chunks_to_fetch.size() == 0) return;  // no missing chunks now

                // 4. Try to get the write lock for the chunks
                bool w_access = try_lock_write(chunks_to_fetch);
                while (!w_access) {
                    cv_.wait(table_lock);
                    chunks_to_fetch.erase(std::remove_if(chunks_to_fetch.begin(), 
                                                      chunks_to_fetch.end(), 
                                                      [this](size_t chunk_id){
                                                          return is_cached_[chunk_id];
                                                      }),
                                         chunks_to_fetch.end());
                    if (chunks_to_fetch.size() == 0) return;  // no missing chunks now
                    w_access = try_lock_write(chunks_to_fetch);
                }

            }

            // 5. Issue fetch_chunk concurrently
            auto ts = fetch_chunk(chunks_to_fetch, local_id);
            husky::LOG_I << "fetch " << chunks_to_fetch.size() << " chunks";
            wait(ts, local_id);

            {
                // 6. Release the write lock
                std::lock_guard<std::mutex> table_lock(lock_table_mtx_);
                // update is_cached_
                for (auto chunk_id : chunks_to_fetch) is_cached_[chunk_id] = true;
                unlock_write(chunks_to_fetch);
            }
        }
    }

    /*
     * Acquire the wirte lock to load the chunks from kvstore
     */
    bool try_lock_write(const std::vector<size_t>& w_chunks) {  // called when lock_table_ is locked
        for (auto w : w_chunks) {
            if (lock_table_[w] != 0) return false;
        }
        // lock chunks
        for (auto w : w_chunks) { lock_table_[w] = -1; }
        return true;
    }

    /*
     * Release the write lock
     */
    void unlock_write(const std::vector<size_t>& chunks) {
        for (auto chunk : chunks) {
            lock_table_[chunk] = 0;
        }
        cv_.notify_all();
    }

    std::mutex lock_table_mtx_;  // for the lock table
    std::vector<int> lock_table_;  // 0 for no lock, positive for read lock, -1 for write lock
    std::condition_variable cv_;
};


/*
 * ChunkBasedMTLockModel
 *
 * ChunkBasedMTLockModel with Read/Write lock for each chunks for multi-threads
 *
 * TODO: handle writer starvation (prioritize write or read?)
 */
class ChunkBasedMTLockModel : public ChunkBasedMTModel {
   public:
    ChunkBasedMTLockModel(int model_id, int num_params):
        ChunkBasedMTModel(model_id, num_params) {} 

    /*
     * Update locally by chunks
     * 1. Acquire locks atomically
     *     1.1 lock lock_table
     *     1.2 wait if any lock cannot be acquired
     *     1.3 else lock all chunks requested and unlock lock_table
     * 2. Do the updates
     *     2.1 find the chunk and index
     *     2.2 update
     * 3. Unlock chunks
     *     3.1 lock lock_table
     *     3.2 unlock chunks
     *     3.3 unlock lock_table and notify all
     */
    void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        auto& range_manager = kvstore::RangeManager::Get();
        // Get chunk indices 
        std::vector<size_t> chunks;
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            chunks.push_back(loc.first);
        }

        // 1. Acquire the lock
        {
            std::unique_lock<std::mutex> table_lock(lock_table_mtx_);
            bool available = try_lock_write(chunks);
            while (!available) {
                cv_.wait(table_lock);
                available = try_lock_write(chunks);
            }
        }

        // 2. Do the updates
        ChunkBasedModel::Push(keys, vals);

        {
            // 3. Unlock the chunks
            std::lock_guard<std::mutex> table_lock(lock_table_mtx_);
            unlock_write(chunks);
        }
    }

    /* Get local cache and fetch on miss
     * 1. Acquire locks atomically
     *     1.1 lock lock_table
     *     1.2 wait if any lock cannot be acquired
     *     1.3 else lock all chunks requested and unlock lock_table
     * 2. Get chunks
     *     2.1 find the chunk and index
     *     2.2 copy
     * 3. Unlock chunks
     *     3.1 lock lock_table
     *     3.2 unlock chunks
     *     3.3 unlock lock_table
     */
    void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals, int local_id) override {
        // Prepare the keys
        Prepare(keys, local_id);

        auto& range_manager = kvstore::RangeManager::Get();
        std::vector<size_t> chunks;
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            chunks.push_back(loc.first);
        }

        // 1. Acquire the read locks
        {
            std::unique_lock<std::mutex> table_lock(lock_table_mtx_);
            bool r_access = try_lock_read(chunks);
            while (!r_access) {
                cv_.wait(table_lock);
                r_access = try_lock_read(chunks);
            }
        }

        // 2. Get the chunks
        vals->resize(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = kvstore::RangeManager::Get().GetLocation(model_id_, keys[i]);
            (*vals)[i] = params_[loc.first][loc.second];
        }

        {
            // 3. Unlock the chunks
            std::unique_lock<std::mutex> table_lock(lock_table_mtx_);
            unlock_read(chunks);
        }
    }

   protected:
    /*
     * Try to get the read lock for the chunks
     */
    bool try_lock_read(const std::vector<size_t>& chunks) {
        for (auto chunk_id : chunks) {
            if (lock_table_[chunk_id] == -1) return false;
        }

        // lock chunks
        for (auto chunk_id : chunks) { lock_table_[chunk_id] += 1; }
        return true;
    }

    /*
     * Release the read lock for the chunks
     */
    void unlock_read(const std::vector<size_t>& chunks) {
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
