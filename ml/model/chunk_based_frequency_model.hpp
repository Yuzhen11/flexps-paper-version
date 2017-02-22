#pragma once

#include <unordered_map>
#include <vector>

#include "kvstore/kvstore.hpp"
#include "ml/model/chunk_based_model.hpp"
#include "ml/model/chunk_based_mt_model.hpp"

namespace ml {
namespace model {

class ChunkBasedFrequencyModel : public ChunkBasedModel {
   public:
    ChunkBasedFrequencyModel(int model_id, int num_params):
        ChunkBasedModel(model_id, num_params) {}

    void LoadFrequent(int local_id, const std::vector<husky::constants::Key>& frequent_ids) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
        std::vector<float> params;
        int ts = kvworker->Pull(model_id_, frequent_ids, &params);
        kvworker->Wait(model_id_, ts);
        for (size_t i = 0; i < frequent_ids.size(); ++i) {
            frequent_pool_[frequent_ids[i]] = params[i];
        }
    }

    void Dump(int local_id, const std::string& hint) override {
        // 1. Dump all chunks
        DumpAllChunks(local_id, model_id_, params_);

        // 2. Dump frequent parameters
        auto size = frequent_pool_.size();
        std::vector<float> vals(size);
        std::vector<husky::constants::Key> keys(size);
        size_t index = 0;
        for (auto f : frequent_pool_) {
            vals[index] = f.second;
            keys[index] = f.first;
            ++index;
        }
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
        auto ts = kvworker->Push(model_id_, keys, vals);
        kvworker->Wait(model_id_, ts);
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        auto& range_manager = kvstore::RangeManager::Get();
        for (size_t i = 0; i < keys.size(); ++i) {
            // 1. find the key in frequent pool
            auto iter = frequent_pool_.find(keys[i]);
            if (iter != frequent_pool_.end()) {
                iter->second += vals[i];
            } 
            // 2. if miss, find it in the chunks
            else {
                auto loc = range_manager.GetLocation(model_id_, keys[i]);
                params_[loc.first][loc.second] += vals[i];
            }
        }
    }

    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals, int local_id) override {
        Prepare(keys, local_id);
        vals->resize(keys.size());
        auto& range_manager = kvstore::RangeManager::Get();
        for (size_t i = 0; i < keys.size(); ++i) {
            auto iter = frequent_pool_.find(keys[i]);
            if (iter != frequent_pool_.end()) {
                vals->at(i) = iter->second;
            } else {
                auto loc = range_manager.GetLocation(model_id_, keys[i]);
                vals->at(i) = params_[loc.first][loc.second];
            }
        }
    }

   protected:
    virtual void Prepare(const std::vector<husky::constants::Key>& keys, int local_id) override {
        auto& range_manager = kvstore::RangeManager::Get();
        std::vector<size_t> chunks_to_fetch;
        for (size_t i = 0; i < keys.size(); ++i) {
            auto iter = frequent_pool_.find(keys[i]);
            if (iter == frequent_pool_.end()) {
                auto loc = range_manager.GetLocation(model_id_, keys[i]);
                if (is_cached_[loc.first] == false && (chunks_to_fetch.empty() || loc.first != chunks_to_fetch.back())) {
                    chunks_to_fetch.push_back(loc.first);
                    is_cached_[loc.first] = true;
                }
            }
        }
        if (!chunks_to_fetch.empty()) {
            int ts = fetch_chunk(chunks_to_fetch, local_id);
            wait(ts, local_id);
        }
    }

    std::unordered_map<husky::constants::Key, float> frequent_pool_;
};

class ChunkBasedMTFrequencyModel : public ChunkBasedFrequencyModel {
   public:
    ChunkBasedMTFrequencyModel(int model_id, int num_params):
        ChunkBasedFrequencyModel(model_id, num_params),
        chunk_lock_(kvstore::RangeManager::Get().GetChunkNum(model_id)) {}

   protected:
    virtual void Prepare(const std::vector<husky::constants::Key>& keys, int local_id) override {
        auto& range_manager = kvstore::RangeManager::Get();
        std::vector<size_t> chunks_to_fetch;
        // 1. Collect the uncached chunks
        for (size_t i = 0; i < keys.size(); ++i) {
            auto iter = frequent_pool_.find(keys[i]);
            if (iter == frequent_pool_.end()) {
                auto loc = range_manager.GetLocation(model_id_, keys[i]);
                if (is_cached_[loc.first] == false && (chunks_to_fetch.empty() || loc.first != chunks_to_fetch.back())) {
                    chunks_to_fetch.push_back(loc.first);
                }
            }
        }

        // 2. Test the locking criterion without actually acquiring the lock
        //    If there is any missing chunk, lock and pull with kvworker
        if (chunks_to_fetch.size() > 0) {
            {
                // 3. Lock and check again
                std::unique_lock<std::mutex> table_lock(chunk_lock_.lock_table_mtx_);

                // Erase-remove idiom
                chunks_to_fetch.erase(std::remove_if(chunks_to_fetch.begin(), 
                                                  chunks_to_fetch.end(), 
                                                  [this](size_t chunk_id){
                                                      return is_cached_[chunk_id];
                                                  }),
                                     chunks_to_fetch.end());
                if (chunks_to_fetch.size() == 0) return;  // no missing chunks now

                // 4. Try to get the write lock for the chunks
                bool w_access = chunk_lock_.try_lock_write(chunks_to_fetch);
                while (!w_access) {
                    chunk_lock_.cv_.wait(table_lock);
                    chunks_to_fetch.erase(std::remove_if(chunks_to_fetch.begin(), 
                                                      chunks_to_fetch.end(), 
                                                      [this](size_t chunk_id){
                                                          return is_cached_[chunk_id];
                                                      }),
                                         chunks_to_fetch.end());
                    if (chunks_to_fetch.size() == 0) return;  // no missing chunks now
                    w_access = chunk_lock_.try_lock_write(chunks_to_fetch);
                }

            }

            // 5. Issue fetch_chunk concurrently
            auto ts = fetch_chunk(chunks_to_fetch, local_id);
            wait(ts, local_id);

            {
                // 6. Release the write lock
                std::lock_guard<std::mutex> table_lock(chunk_lock_.lock_table_mtx_);
                // update is_cached_
                for (auto chunk_id : chunks_to_fetch) is_cached_[chunk_id] = true;
                chunk_lock_.unlock_write(chunks_to_fetch);
            }
        }
    }

    ChunkLock chunk_lock_;
};

class ChunkBasedMTLockFrequencyModel : public ChunkBasedMTFrequencyModel {
   public:
    ChunkBasedMTLockFrequencyModel(int model_id, int num_params):
        ChunkBasedMTFrequencyModel(model_id, num_params) {}

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
            std::unique_lock<std::mutex> table_lock(chunk_lock_.lock_table_mtx_);
            bool available = chunk_lock_.try_lock_write(chunks);
            while (!available) {
                chunk_lock_.cv_.wait(table_lock);
                available = chunk_lock_.try_lock_write(chunks);
            }
        }

        // 2. Do the updates
        ChunkBasedFrequencyModel::Push(keys, vals);

        {
            // 3. Unlock the chunks
            std::lock_guard<std::mutex> table_lock(chunk_lock_.lock_table_mtx_);
            chunk_lock_.unlock_write(chunks);
        }
    }

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
            std::unique_lock<std::mutex> table_lock(chunk_lock_.lock_table_mtx_);
            bool r_access = chunk_lock_.try_lock_read(chunks);
            while (!r_access) {
                chunk_lock_.cv_.wait(table_lock);
                r_access = chunk_lock_.try_lock_read(chunks);
            }
        }

        // 2. Get the chunks
        vals->resize(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            auto iter = frequent_pool_.find(keys[i]);
            if (iter != frequent_pool_.end()) {
                vals->at(i) = iter->second;
            } else {
                auto loc = range_manager.GetLocation(model_id_, keys[i]);
                (*vals)[i] = params_[loc.first][loc.second];
            }
        }

        {
            // 3. Unlock the chunks
            std::unique_lock<std::mutex> table_lock(chunk_lock_.lock_table_mtx_);
            chunk_lock_.unlock_read(chunks);
        }
    }
};

}  // namespace model
}  // namespace ml
