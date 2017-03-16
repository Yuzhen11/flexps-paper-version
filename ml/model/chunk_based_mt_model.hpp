#pragma once

#include <cassert>
#include <vector>

#include "boost/thread/shared_mutex.hpp"
#include "boost/thread/locks.hpp"

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
 */
template<typename Val>
class ChunkBasedMTModel : public ChunkBasedModel<Val> {
   public:
    using Model<Val>::model_id_;
    using ChunkBasedModel<Val>::is_cached_;

    ChunkBasedMTModel() = delete;
    ChunkBasedMTModel(int model_id, int num_params):
        ChunkBasedModel<Val>(model_id, num_params),
        mtx_(kvstore::RangeManager::Get().GetChunkNum(model_id)) {}

    /*
     * Override the Prepare function in ChunkBasedModel
     *
     * Allow concurrent Pull issues to kvstore
     * A chunk is locked only once and then in cache forever, so only fetch those not in cache and not locked
     */
    virtual void Prepare(const std::vector<husky::constants::Key>& keys, int local_id) override {
        std::vector<size_t> chunks_to_fetch;
        std::vector<size_t> chunks_to_check;

        // 1. Collect the uncached chunks
        size_t current_chunk_id;
        auto& range_manager = kvstore::RangeManager::Get();
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            auto chunk_id = loc.first;
            if (is_cached_[chunk_id] == false && (i == 0 || chunk_id != current_chunk_id)) {
                // try to get the write lock for the chunk
                // if success prepare to pull it from kvstore
                if (mtx_[chunk_id].try_lock()) {
                    if (is_cached_[chunk_id] == false) {
                        chunks_to_fetch.push_back(chunk_id);
                    } else {
                        mtx_[chunk_id].unlock();
                    }
                } else {  // locked by other threads and is being fetched
                    chunks_to_check.push_back(chunk_id);
                }
            }
            current_chunk_id = chunk_id;
        }

        // 2. Pull chunks_to_fetch from kvstore
        if (chunks_to_fetch.size() > 0) {
            auto ts = ChunkBasedModel<Val>::fetch_chunk(chunks_to_fetch, local_id);
            ChunkBasedModel<Val>::wait(ts, local_id);
            // 3. Unlock the mutices
            for (auto chunk_id : chunks_to_fetch) {
                is_cached_[chunk_id] = true;
                mtx_[chunk_id].unlock();
            }
        }

        // 4. Wait until chunks_to_check are in cache now
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


/*
 * ChunkBasedMTLockModel
 *
 * ChunkBasedMTLockModel with Read/Write lock for each chunks for multi-threads
 *
 * TODO: handle writer starvation (prioritize write or read?)
 */
template<typename Val>
class ChunkBasedMTLockModel : public ChunkBasedMTModel<Val> {
   public:
    using Model<Val>::model_id_;
    using ChunkBasedModel<Val>::params_;
    using ChunkBasedMTModel<Val>::mtx_;

    ChunkBasedMTLockModel(int model_id, int num_params):
        ChunkBasedMTModel<Val>(model_id, num_params) {}

    /*
     * Update locally by chunks
     * Override the Push function in ChunkBasedModel
     * TODO use try_lock and continue with next chunk without blocking?
     */
    void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) override {
        if (keys.empty()) return;
        auto& range_manager = kvstore::RangeManager::Get();

        size_t current_chunk_id;
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
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
            // Update the parameter
            params_[chunk_id][loc.second] += vals[i];
        }
        mtx_[current_chunk_id].unlock();
    }

    /* Get local cache and pull from kvstore on miss
     * Override the Pull function in ChunkBasedModel
     */
    void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals, int local_id) override {
        if (keys.empty()) return;
        // Prepare the keys
        ChunkBasedMTModel<Val>::Prepare(keys, local_id);

        vals->resize(keys.size());

        auto& range_manager = kvstore::RangeManager::Get();
        size_t current_chunk_id;
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            auto chunk_id = loc.first;
            if (i == 0 || chunk_id != current_chunk_id) {
                if (i != 0) {
                    // Unlock the last chunk
                    mtx_[current_chunk_id].unlock();
                }
                // Acquire read lock
                mtx_[chunk_id].lock();
                current_chunk_id = chunk_id;
            }
            vals->at(i) = params_[chunk_id][loc.second];
        }
        mtx_[current_chunk_id].unlock();
    }
};

}  // namespace model
}  // namespace ml
