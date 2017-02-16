#pragma once

#include <cassert>
#include <vector>

#include "core/constants.hpp"
#include "ml/model/model.hpp"
#include "kvstore/kvstore.hpp"

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
class ChunkBasedModel : public Model {
   public:
    ChunkBasedModel(int model_id, int num_params):
        Model(model_id, num_params),
        params_(kvstore::RangeManager::Get().GetChunkNum(model_id)),
        is_cached_(kvstore::RangeManager::Get().GetChunkNum(model_id), false) {}

    void Load(int local_id) override {}
    void Dump(int local_id) override {
        // TODO
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        // chunks must be pulled before push
        auto& range_manager = kvstore::RangeManager::Get();
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            assert(is_cached_[loc.first]);
            params_[loc.first][loc.second] += vals[i];
        }
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals, int local_id) override {
        Prepare(keys, local_id);
        vals->resize(keys.size());
        auto& range_manager = kvstore::RangeManager::Get();
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            (*vals)[i] = params_[loc.first][loc.second];
        }
    }

   protected:
    /*
     * Prepare function to prepare the chunks for the provided keys
     *
     * For single thread, no need to lock the chunks I am preparing
     */
    virtual void Prepare(const std::vector<husky::constants::Key>& keys, int local_id) {
        auto& range_manager = kvstore::RangeManager::Get();
        // The keys should be in ascending order
        std::vector<size_t> chunks_to_fetch;
        for (size_t i = 0; i < keys.size(); ++ i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            if (is_cached_[loc.first] == false) {
                chunks_to_fetch.push_back(loc.first);
                is_cached_[loc.first] = true;
            }
        }
        int ts = fetch_chunk(chunks_to_fetch, local_id);
        husky::LOG_I << "fetch " << chunks_to_fetch.size() << " chunks";
        wait(ts, local_id);
    }

    /*
     * Fetch the given chunks, return a timestamp
     */
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

    /*
     * Wait for the timestamp
     */
    void wait(int ts, int local_id) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
        kvworker->Wait(this->model_id_, ts);
    }

    std::vector<std::vector<float>> params_;  // params in chunks
    std::vector<bool> is_cached_;  // indicate whether chunk has been pulled from kvstore
};
}  // namespace model
}  // namespace ml
