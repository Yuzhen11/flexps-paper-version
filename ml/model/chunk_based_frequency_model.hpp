#pragma once

#include <unordered_map>
#include <vector>

#include "boost/thread/mutex.hpp"
#include "boost/thread/shared_mutex.hpp"
#include "boost/thread/locks.hpp"

#include "kvstore/kvstore.hpp"
#include "ml/model/chunk_based_model.hpp"
#include "ml/model/chunk_based_mt_model.hpp"

namespace ml {
namespace model {

template<typename Val>
class ChunkBasedFrequencyModel : public ChunkBasedModel<Val> {
   public:
    using ChunkBasedModel<Val>::model_id_;
    using ChunkBasedModel<Val>::params_;
    using ChunkBasedModel<Val>::is_cached_;

    ChunkBasedFrequencyModel(int model_id, int num_params):
        ChunkBasedModel<Val>(model_id, num_params) {}

    void LoadFrequent(int local_id, const std::vector<husky::constants::Key>& frequent_ids) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
        std::vector<Val> params;
        int ts = kvworker->Pull(model_id_, frequent_ids, &params);
        kvworker->Wait(model_id_, ts);
        for (size_t i = 0; i < frequent_ids.size(); ++i) {
            frequent_pool_[frequent_ids[i]] = params[i];
        }
    }

    void Dump(int local_id, const std::string& hint) override {
        // 1. Dump all chunks
        DumpAllChunksToKV(local_id, model_id_, params_);

        // 2. Dump frequent parameters
        auto size = frequent_pool_.size();
        std::vector<Val> vals(size);
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

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) override {
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

    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals, int local_id) override {
        Prepare(keys, local_id);
        vals->resize(keys.size());
        auto& range_manager = kvstore::RangeManager::Get();
        for (size_t i = 0; i < keys.size(); ++i) {
            auto iter = frequent_pool_.find(keys[i]);
            if (iter != frequent_pool_.end()) {
                (*vals)[i] = iter->second;
            } else {
                auto loc = range_manager.GetLocation(model_id_, keys[i]);
                (*vals)[i] = params_[loc.first][loc.second];
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
            int ts = this->fetch_chunk(chunks_to_fetch, local_id);
            ChunkBasedModel<Val>::wait(ts, local_id);
        }
    }

    std::unordered_map<husky::constants::Key, Val> frequent_pool_;
};

template<typename Val>
class ChunkBasedMTFrequencyModel : public ChunkBasedFrequencyModel<Val> {
   public:
    using ChunkBasedFrequencyModel<Val>::frequent_pool_;
    using ChunkBasedModel<Val>::model_id_;
    using ChunkBasedModel<Val>::params_;
    using ChunkBasedModel<Val>::is_cached_;

    ChunkBasedMTFrequencyModel(int model_id, int num_params):
        ChunkBasedFrequencyModel<float>(model_id, num_params),
        mtx_(kvstore::RangeManager::Get().GetChunkNum(model_id)) {}

    virtual void Prepare(const std::vector<husky::constants::Key>& keys, int local_id) override {
        auto& range_manager = kvstore::RangeManager::Get();
        std::vector<size_t> chunks_to_fetch;
        std::vector<size_t> chunks_to_check;

        // 1. Collect the uncached chunks
        for (size_t i = 0; i < keys.size(); ++i) {
            auto iter = frequent_pool_.find(keys[i]);
            if (iter == frequent_pool_.end()) {
                auto loc = range_manager.GetLocation(model_id_, keys[i]);
                auto chunk_id = loc.first;
                if (is_cached_[chunk_id] == false && (chunks_to_fetch.empty() || chunk_id != chunks_to_fetch.back()) && (chunks_to_check.empty() || chunk_id != chunks_to_check.back())) {
                    if (mtx_[chunk_id].try_lock()) {
                        if (is_cached_[chunk_id]) {
                            mtx_[chunk_id].unlock();
                        } else {
                            chunks_to_fetch.push_back(chunk_id);
                        }
                    } else {
                        chunks_to_check.push_back(chunk_id);
                    }
                }
            }
        }

        if (!chunks_to_fetch.empty()) {
            auto ts = this->fetch_chunk(chunks_to_fetch, local_id);
            ChunkBasedModel<Val>::wait(ts, local_id);
            for (auto chunk_id : chunks_to_fetch) {
                assert(params_[chunk_id].size() > 0);  // for debug
                is_cached_[chunk_id] = true;
                mtx_[chunk_id].unlock();
            }
        }

        if (!chunks_to_check.empty()) {
            chunks_to_check.erase(std::remove_if(chunks_to_check.begin(), chunks_to_check.end(),
                        [this](size_t chunk_id) { return is_cached_[chunk_id]; }),
                    chunks_to_check.end());
            if (!chunks_to_check.empty()) {
                for (auto chunk_id : chunks_to_check) {
                    mtx_[chunk_id].lock();
                    assert(is_cached_[chunk_id]);  // for debug
                    mtx_[chunk_id].unlock();
                }
            }
        }
    }

   protected:
    std::vector<boost::mutex> mtx_;
};

template<typename Val>
class ChunkBasedMTLockFrequencyModel : public ChunkBasedMTFrequencyModel<Val> {
   public:
    using ChunkBasedFrequencyModel<Val>::frequent_pool_;
    using ChunkBasedModel<Val>::model_id_;
    using ChunkBasedModel<Val>::params_;
    using ChunkBasedModel<Val>::is_cached_;
    using ChunkBasedMTFrequencyModel<Val>::mtx_;

    ChunkBasedMTLockFrequencyModel(int model_id, int num_params):
        ChunkBasedMTFrequencyModel<Val>(model_id, num_params) {}

    void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) override {
        auto& range_manager = kvstore::RangeManager::Get();

        size_t current_chunk_id;
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            // 1. Get write lock on the current chunk
            auto chunk_id = loc.first;
            // If this is a different chunk ...
            if (i == 0 || chunk_id != current_chunk_id) {
                if (i != 0) {
                    // Unlock the last chunk
                    mtx_[current_chunk_id].unlock();
                }
                // Acquire write lock
                mtx_[chunk_id].lock();
                current_chunk_id = chunk_id;
            }
            // 2. Update the parameter
            // Find the key in frequent pool
            auto iter = frequent_pool_.find(keys[i]);
            if (iter != frequent_pool_.end()) {
                iter->second += vals[i];
            } else {
                params_[chunk_id][loc.second] += vals[i];
            }
        }
        mtx_[current_chunk_id].unlock();
    }

    void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals, int local_id) override {
        // Prepare the keys
        this->Prepare(keys, local_id);

        auto& range_manager = kvstore::RangeManager::Get();
        vals->resize(keys.size());
        size_t current_chunk_id;
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            // 1. Get read lock on the current chunk
            auto chunk_id = loc.first;
            // If this is a different chunk ...
            if (i == 0 || chunk_id != current_chunk_id) {
                if (i != 0) {
                    // Unlock the last chunk
                    mtx_[current_chunk_id].unlock();
                }
                // Acquire write lock
                mtx_[chunk_id].lock();
                current_chunk_id = chunk_id;
            }
            // 2. Copy to vals
            // Find the key in frequent pool
            auto iter = frequent_pool_.find(keys[i]);
            if (iter != frequent_pool_.end()) {
                (*vals)[i] = iter->second;
            } else {
                (*vals)[i] = params_[loc.first][loc.second];
            }
        }
        mtx_[current_chunk_id].unlock();
    }
};

}  // namespace model
}  // namespace ml
