#pragma once

#include <vector>

#include "boost/iterator/indirect_iterator.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/thread/locks.hpp"
#include "core/constants.hpp"
#include "kvstore/kvstore.hpp"
#include "ml/model/chunk_based_model.hpp"

namespace ml {
namespace model {

class ChunkBasedPSModel {
   public:
    ChunkBasedPSModel(int model_id, int num_params):
        model_id_(model_id), num_params_(num_params),
        num_chunks_(kvstore::RangeManager::Get().GetChunkNum(model_id)),
        params_(num_chunks_), chunk_clocks_(num_chunks_, -1), mtx_(num_chunks_) {}

    virtual int PullWithMinClock(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals, int local_id, int min_clock) {
        // Prepare the keys
        Prepare(keys, local_id, min_clock);

        int clock = 0;
        vals->resize(keys.size());

        auto& range_manager = kvstore::RangeManager::Get();
        size_t current_chunk_id;
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            auto chunk_id = loc.first;
            if (i == 0 || chunk_id != current_chunk_id) {
                if (i != 0) mtx_[current_chunk_id].unlock();  // unlock the current chunk
                mtx_[chunk_id].lock();  // lock the new chunk
                current_chunk_id = chunk_id;
                if (chunk_clocks_[chunk_id] < clock) clock = chunk_clocks_[chunk_id];
            }
            vals->at(i) = params_[chunk_id][loc.second];
        }
        if (keys.size() > 0) {
            mtx_[current_chunk_id].unlock();  // unlock the last chunk
        }
        
        return clock;
    }

    virtual void PullChunksWithMinClock(std::vector<size_t>& chunks, std::vector<std::vector<float>*>& chunk_ptrs, std::vector<int>& chunk_clocks, int local_id, int min_clock) {
        // Collect chunks that are too stale
        std::vector<size_t> chunks_to_fetch;
        for (auto chunk_id : chunks) {
            if (chunk_clocks_[chunk_id] < min_clock) {
                chunks_to_fetch.push_back(chunk_id);
            }
        }

        // Pull from kvstore
        if (!chunks_to_fetch.empty()) {
            fetch_chunk(chunks_to_fetch, local_id);
        }

        for (int i = 0; i < chunks.size(); ++i) {
            mtx_[chunks[i]].lock();
            *chunk_ptrs[i] = params_[chunks[i]];
            chunk_clocks[chunks[i]] = chunk_clocks_[chunks[i]];
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

   protected:
    void fetch_chunk(const std::vector<size_t>& chunks, int local_id) {
        int clock = 0;
        // 1. get kvworker
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);

        // 2. pull chunks
        std::vector<std::vector<float>*> chunk_ptrs;
        chunk_ptrs.reserve(chunks.size());
        std::vector<std::vector<float>> tmp_chunks(chunks.size());
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
    }

    int model_id_;
    int num_params_;
    int num_chunks_;
    std::vector<std::vector<float>> params_;
    std::vector<int> chunk_clocks_;
    std::vector<boost::mutex> mtx_;
};

class ChunkBasedModelWithClocks : public ChunkBasedModel {
   public:
    ChunkBasedModelWithClocks(int model_id, int num_params) :
        ChunkBasedModel(model_id, num_params),
        chunk_clocks_(num_chunks_, -1),
        range_manager_(&kvstore::RangeManager::Get()) {}

    inline void SetStaleness(int staleness) { staleness_ = staleness; }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        // 1. Update local model
        ChunkBasedModel::Push(keys, vals);

        // 2. Inc clock
        ++clock_;
    }

    virtual void Prepare(const std::vector<husky::constants::Key>& keys, int local_id) override {
        // The keys should be in ascending order
        std::vector<size_t> chunks_to_fetch;
        int stalest = clock_ - staleness_;
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager_->GetLocation(model_id_, keys[i]);
            if ((is_cached_[loc.first] == false || chunk_clocks_[loc.first] < stalest) && (chunks_to_fetch.empty() || loc.first != chunks_to_fetch.back())) {
                chunks_to_fetch.push_back(loc.first);
                is_cached_[loc.first] = true;
            }
        }
        if (chunks_to_fetch.empty()) return;
        fetch_chunk(chunks_to_fetch, local_id);
    }


    float At(const husky::constants::Key& idx) {
        auto loc = range_manager_->GetLocation(model_id_, idx);
        return params_[loc.first][loc.second];
    }

    void Inc(const husky::constants::Key& idx, const float& val) {
        auto loc = range_manager_->GetLocation(model_id_, idx);
        params_[loc.first][loc.second] += val;
    }

   private:
    int fetch_chunk(const std::vector<size_t>& chunks, int local_id) override {
        // 1. Get kvworker
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);

        // 2. Pull Chunks
        std::vector<std::vector<float>*> chunk_ptrs(chunks.size());
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

}  // namespace model
}  // namespace ml
