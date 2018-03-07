#pragma once

#include <cassert>
#include <vector>

#include "core/constants.hpp"
#include "kvstore/kvstore.hpp"
#include "ml/model/load.hpp"
#include "ml/model/dump.hpp"
#include "ml/model/model.hpp"

namespace ml {
namespace model {

/*
 * ChunkBasedModel
 *
 * The most basic Chunk-based Model.
 *
 * ChunkBased Model for Single. Since there's only one thread,
 * no need to use double checked locking to allow the concurrecny Pull operations to kvstore
 */
template<typename Val>
class ChunkBasedModel : public Model<Val> {
   public:
    using Model<Val>::model_id_;

    ChunkBasedModel(int model_id, int num_params):
        Model<Val>(model_id, num_params),
        num_chunks_(kvstore::RangeManager::Get().GetChunkNum(model_id)),
        params_(num_chunks_),
        is_cached_(num_chunks_, false) {}

    void Load(int local_id, int task_id, const std::string& hint) override {
        if (hint == husky::constants::kKVStoreChunks) {
            // Do nothing
        } else if (hint == husky::constants::kKVStoreIntegral) {
            // Load all from KVStore 
            LoadAllChunksFromKV(local_id, model_id_, &params_);
            std::fill(is_cached_.begin(), is_cached_.end(), 1);
        } else if (hint == husky::constants::kTransferIntegral) {
            // Load from kTransferChunks
            LoadAllChunksFromStore(local_id, task_id, &params_);
            std::fill(is_cached_.begin(), is_cached_.end(), 1);
        } else {
            throw husky::base::HuskyException("Unknown hint in ChunkModel: "+hint);
        }
    }

    virtual void Dump(int local_id, int task_id, const std::string& hint) override {
        if (hint == husky::constants::kKVStoreChunks) {
            // just need to dump some chunks, if chunk is null, it needn't dumped
            std::vector<std::vector<Val>*> chunks;
            // chunk ids
            std::vector<size_t> chunk_ids;
            for (size_t i = 0; i < params_.size(); i++) {
                if (params_[i].size()) {
                    chunk_ids.push_back(i);
                    chunks.push_back(&params_[i]);
                }
            }
            DumpChunks(local_id, model_id_, chunk_ids, chunks);
        } else if (hint == husky::constants::kKVStoreIntegral) {
            DumpAllChunksToKV(local_id, model_id_, params_);
        } else if (hint == husky::constants::kTransferIntegral) {
            DumpAllChunksToStore(task_id, params_);
        } else {
            throw husky::base::HuskyException("Unknown hint in ChunkModel: "+hint);
        }
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) override {
        // chunks must be pulled before push
        auto& range_manager = kvstore::RangeManager::Get();
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            assert(is_cached_[loc.first]);
            params_[loc.first][loc.second] += vals[i];
        }
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals, int local_id) override {
        Prepare(keys, local_id);
        vals->resize(keys.size());
        auto& range_manager = kvstore::RangeManager::Get();
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            (*vals)[i] = params_[loc.first][loc.second];
        }
    }
    virtual void PushChunks(const std::vector<size_t>& chunk_keys, const std::vector<std::vector<Val>*>& chunk_vals) override {
        for (size_t i = 0; i < chunk_keys.size(); ++ i) {
            size_t chunk_id = chunk_keys[i];
            assert(params_[chunk_id].size() == chunk_vals[i]->size());
            for (size_t j = 0; j < chunk_vals[i]->size(); ++ j) {
                params_[chunk_id][j] += (*(chunk_vals[i]))[j];
            }
        }
    }
    void PullChunks(const std::vector<size_t>& chunk_keys, std::vector<std::vector<Val>*>& chunk_vals, int local_id) override {
        if (chunk_keys.empty()) return;
        assert(chunk_keys.size() == chunk_vals.size());
        PrepareChunks(chunk_keys, local_id);
        for (size_t i = 0; i < chunk_keys.size(); i++) {
            size_t chunk_id = chunk_keys[i];
            chunk_vals[i]->resize(params_[chunk_id].size());
            for (size_t j = 0; j < chunk_vals[i]->size(); j++) {
                (*(chunk_vals[i]))[j] = params_[chunk_id][j];
            }
        }
    }

    /*
     * Prepare function to prepare the chunks for the provided keys
     *
     * For single thread, no need to lock the chunks I am preparing
     *
     * 1. Get the chunk_ids
     * 2. Call the PrepareChunks (virtual)
     */
    virtual void Prepare(const std::vector<husky::constants::Key>& keys, int local_id) {
        std::vector<size_t> chunk_ids;
        auto& range_manager = kvstore::RangeManager::Get();
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            if (chunk_ids.empty() || loc.first != chunk_ids.back()) {
                chunk_ids.push_back(loc.first);
            }
        }
        PrepareChunks(chunk_ids, local_id);
    }

    virtual void PrepareChunks(const std::vector<size_t>& chunk_keys, int local_id) {
        if (chunk_keys.empty()) return;
        std::vector<size_t> chunks_to_fetch;
        for (auto chunk_key : chunk_keys) {
            assert(chunk_key < is_cached_.size());
            if (is_cached_[chunk_key] == false) {
                chunks_to_fetch.push_back(chunk_key);
                is_cached_[chunk_key] = true;
            }
        }
        if (chunks_to_fetch.empty())
            return;
        int ts = fetch_chunk(chunks_to_fetch, local_id);
        wait(ts, local_id);
    }

    /*
     * Get the raw pointer to the params_
     */
    std::vector<std::vector<Val>>* GetParamsPtr() {
        return &params_;
    }
   protected:
    /*
     * Fetch the given chunks, return a timestamp
     */
    virtual int fetch_chunk(const std::vector<size_t>& chunks, int local_id) {
        assert(chunks.size() > 0);
        // 1. get kvworker
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
        // 2. pull chunks and return ts
        std::vector<std::vector<Val>*> chunk_ptrs;
        chunk_ptrs.reserve(chunks.size());
        for (auto chunk_id : chunks) {
            chunk_ptrs.push_back(&(params_[chunk_id]));
        }
        return kvworker->PullChunks(this->model_id_, chunks, chunk_ptrs, false);
    }

    /*
     * Wait for the timestamp
     */
    void wait(int ts, int local_id) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
        kvworker->Wait(this->model_id_, ts);
    }

    int num_chunks_;
    std::vector<std::vector<Val>> params_;  // params in chunks
    std::vector<bool> is_cached_;  // indicate whether chunk has been pulled from kvstore
};

}  // namespace model
}  // namespace ml
