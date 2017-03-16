#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <unordered_map>
#include <utility>
#include <stdio.h>
#include <vector>

#include "boost/iterator/indirect_iterator.hpp"
#include "core/constants.hpp"
#include "ml/model/chunk_based_mt_model.hpp"
#include "kvstore/kvstore.hpp"

namespace ml {
namespace model {

enum CacheStatus { InKVStore = 0, InDisk = 1 };

template<typename Val>
class ChunkFileEditor {
   public:
    ChunkFileEditor(std::vector<std::vector<Val>>* params, int chunk_size, int last_chunk_size, int num_chunks):
        chunks_ptr_(params), chunk_size_(chunk_size), last_chunk_size_(last_chunk_size), num_chunks_(num_chunks) {
            file_ = tmpfile();
            if (file_ == NULL) {
                throw husky::base::HuskyException("Cannot create tmp file");
            }
        }

    ~ChunkFileEditor() {
        if (file_ != NULL) {
            fclose(file_);
        }
    }

    void read_chunks(const std::vector<size_t>& ids) {
        boost::lock_guard<boost::mutex> lock(mtx_);
        for (auto id : ids) {
            auto iter = chunk_map_.find(id);
            if (iter == chunk_map_.end()) throw husky::base::HuskyException("Target chunk is not in disk");

            // allocate memory to chunk
            int size = chunk_size_;
            if (id == num_chunks_ - 1) size = last_chunk_size_;
            chunks_ptr_->at(id).resize(size);

            fseek(file_, iter->second, SEEK_SET);
            fread(chunks_ptr_->at(id).data(), sizeof(Val), size, file_);
        }
    }

    void write_chunks(const std::vector<size_t>& ids) {
        boost::lock_guard<boost::mutex> lock(mtx_);
        for (auto id : ids) {
            auto iter = chunk_map_.find(id);
            // get actual size of chunk
            int size = chunk_size_;
            if (id == num_chunks_ - 1) size = last_chunk_size_;

            if (iter != chunk_map_.end()) {  // if the chunk is previously written
                fseek(file_, iter->second, SEEK_SET);
                fwrite(chunks_ptr_->at(id).data(), sizeof(Val), size, file_);
            } else {  // first time to write the chunk
                chunk_map_[id] = bytes_written_;
                fseek(file_, 0, SEEK_END);
                fwrite(chunks_ptr_->at(id).data(), sizeof(Val), size, file_);
                bytes_written_ += size * sizeof(Val);
            }
        }
    }

   protected:
    FILE* file_;
    boost::mutex mtx_;
    std::unordered_map<size_t, int> chunk_map_;  // chunk_id, byte position
    std::vector<std::vector<Val>> * chunks_ptr_ = NULL;
    int bytes_written_ = 0;
    int last_chunk_size_;
    int chunk_size_;
    int num_chunks_;
};

template<typename Val>
class ModelWithCM : public ChunkBasedMTLockModel<Val> {
   public:
    using ChunkBasedMTLockModel<Val>::num_chunks_;
    using ChunkBasedMTLockModel<Val>::params_;
    using ChunkBasedMTLockModel<Val>::mtx_;
    using ChunkBasedModel<Val>::is_cached_;
    using Model<Val>::model_id_;
    ModelWithCM(int model_id, int num_params, int cache_threshold = 0) :
        ChunkBasedMTLockModel<Val>(model_id, num_params), cache_threshold_(cache_threshold),
        status_(num_chunks_, 0), prepare_count_(num_chunks_, 0),
        cfe_(&params_, kvstore::RangeManager::Get().GetChunkSize(model_id), kvstore::RangeManager::Get().GetLastChunkSize(model_id), num_chunks_) {}

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

                touch(chunk_id);
                prepare_count_[chunk_id] -= 1;
                current_chunk_id = chunk_id;
            }
            // Update the parameter
            params_[chunk_id][loc.second] += vals[i];
        }
        mtx_[current_chunk_id].unlock();
    }

    virtual void Prepare(const std::vector<husky::constants::Key>& keys, int local_id) override {
        assert(keys.size() <= cache_threshold_);
        std::vector<size_t> chunks_to_fetch;
        std::vector<size_t> chunks_to_prepare;
        std::vector<boost::mutex*> mtx_ptrs;
        chunks_to_prepare.reserve(num_chunks_);
        mtx_ptrs.reserve(num_chunks_);

        size_t current_chunk_id;
        auto& range_manager = kvstore::RangeManager::Get();
        // 1. Get all mutex of chunks to be prepared
        for (size_t i = 0; i < keys.size(); ++i) {
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            auto chunk_id = loc.first;
            if (i == 0 || chunk_id != current_chunk_id) {
                mtx_ptrs.push_back(&mtx_[chunk_id]);
                chunks_to_prepare.push_back(chunk_id);
                current_chunk_id = chunk_id;
            }
        }
        mtx_ptrs.push_back(&global_mtx_);

        // Lock all needed chunks and global mutex
        boost::indirect_iterator<std::vector<boost::mutex*>::iterator> first(mtx_ptrs.begin());
        boost::indirect_iterator<std::vector<boost::mutex*>::iterator> last(mtx_ptrs.end());
        boost::lock(first, last);

        chunks_to_fetch.reserve(chunks_to_prepare.size());
        for (auto chunk_id : chunks_to_prepare) {
            // 2. Increment prepare count
            prepare_count_[chunk_id] += 1;

            if (is_cached_[chunk_id]) {
                mtx_[chunk_id].unlock();
            } else {
                // 3. Get chunks to fetch
                chunks_to_fetch.push_back(chunk_id);
            }
        }

        if (chunks_to_fetch.empty()) {
            global_mtx_.unlock();
            return;
        }

        // 4. Check whether overflow, replace some chunks
        std::vector<size_t> chunks_to_replace;
        int overflow = num_cached_ + chunks_to_fetch.size() - cache_threshold_;
        if (overflow > 0) {
            replace_lock(overflow, chunks_to_replace);
            num_cached_ = cache_threshold_;
        } else {
            num_cached_ += chunks_to_fetch.size();
        }

        global_mtx_.unlock();

        if (!chunks_to_replace.empty()) {
            flush_to_disk(chunks_to_replace);
            for (auto chunk_id : chunks_to_replace) {
                is_cached_[chunk_id] = false;
                status_[chunk_id] = InDisk;
                mtx_[chunk_id].unlock();
            }
        }

        // 5. Fetch chunks
        auto ts = fetch_chunk(chunks_to_fetch, local_id);
        if (ts != -1) {
            ChunkBasedModel<Val>::wait(ts, local_id);
        }

        // 6. Release the write locks
        for (auto chunk_id : chunks_to_fetch) {
            is_cached_[chunk_id] = true;
            mtx_[chunk_id].unlock();
        }
    }

   protected:
    void flush_to_disk(const std::vector<size_t>& chunks_to_flush) {
        cfe_.write_chunks(chunks_to_flush);
        for (auto id : chunks_to_flush) {
            params_[id].clear();
        }
    }

    void read_from_disk(const std::vector<size_t>& chunks) {
        if (chunks.empty()) return;
        cfe_.read_chunks(chunks);
        // report_miss(chunks);
    }

    // virtual void report_miss(const std::vector<size_t>& chunks) {}

    int fetch_chunk(const std::vector<size_t>& chunks, int local_id) override {
        // 1. Get kvworker
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);

        // 2. find the chunks: in disk or kvstore
        std::vector<size_t> chunks_kvstore;
        std::vector<std::vector<Val>*> chunk_ptrs;
        chunks_kvstore.reserve(chunks.size());
        chunk_ptrs.reserve(chunks.size());

        std::vector<size_t> chunks_disk;
        chunks_disk.reserve(chunks.size());

        for (auto id : chunks) {
            if (status_[id] == InKVStore) {
                chunks_kvstore.push_back(id);
                chunk_ptrs.push_back(&(params_[id]));
            } else {
                chunks_disk.push_back(id);
            }
        }

        // 3. Pull chunks from kvstore
        int ts = -1;
        if (!chunk_ptrs.empty()) {
            ts = kvworker->PullChunks(this->model_id_, std::move(chunks_kvstore), chunk_ptrs, false);
        }

        // 4. Read chunks from disk
        read_from_disk(std::move(chunks_disk));

        return ts;
    }

    virtual void replace_lock(int num_to_replace, std::vector<size_t>& chunks_to_replace) {}
    virtual void touch(size_t chunk_id) {}

    boost::mutex global_mtx_;
    ChunkFileEditor<Val> cfe_;
    std::vector<int> status_;
    std::vector<int> prepare_count_;
    int num_cached_ = 0;
    int cache_threshold_;
};

template<typename Val>
class ModelWithCMLRU : public ModelWithCM<Val> {
   public:
    using ChunkBasedModel<Val>::num_chunks_;
    using ChunkBasedModel<Val>::is_cached_;
    using ModelWithCM<Val>::num_cached_;
    using ModelWithCM<Val>::prepare_count_;
    using ChunkBasedMTModel<Val>::mtx_;
    ModelWithCMLRU(int model_id, int num_params, int cache_threshold):
        ModelWithCM<Val>(model_id, num_params, cache_threshold),
        access_count_(0),
        recency_(num_chunks_, 0) {}

   protected:
    void replace_lock(int num_to_replace, std::vector<size_t>& chunks_to_replace) override {
        std::vector<std::pair<size_t, int>> pool;
        pool.reserve(num_cached_);
        while (pool.size() < num_to_replace) {
            for (size_t i = 0; i < num_chunks_; ++i) {
                if (is_cached_[i] && prepare_count_[i] == 0 && mtx_[i].try_lock()) {  // get unprotected unlocked chunks
                    if (is_cached_[i] && prepare_count_[i] == 0) {
                        pool.push_back(std::make_pair(i, recency_[i]));
                    } else {
                        mtx_[i].unlock();
                    }
                }
            }
            if (pool.size() < num_to_replace) {
                std::this_thread::yield();
            }
        }

        // Sort according to recency in ascending order
        std::sort(pool.begin(), pool.end(), [](std::pair<size_t, int> a, std::pair<size_t, int> b) {
            return a.second < b.second;
        });

        // Get only the least recent chunks
        chunks_to_replace.reserve(num_to_replace);
        for (int i = 0; i < num_to_replace; ++i) {
            chunks_to_replace.push_back(pool[i].first);
        }
        // Unlock the other chunks
        if (pool.size() > num_to_replace) {
            for (int i = num_to_replace; i < pool.size(); ++i) {
                mtx_[pool[i].first].unlock();
            }
        }
    }

    void touch(size_t chunk_id) override {
        ++access_count_;
        recency_[chunk_id] = access_count_;
    }

    std::vector<int> recency_;
    std::atomic_int access_count_;
};

template<typename Val>
class ModelWithCMLFU : public ModelWithCM<Val> {
   public:
    using ChunkBasedModel<Val>::num_chunks_;
    using ChunkBasedModel<Val>::is_cached_;
    using ModelWithCM<Val>::num_cached_;
    using ModelWithCM<Val>::prepare_count_;
    using ChunkBasedMTModel<Val>::mtx_;

    ModelWithCMLFU(int model_id, int num_params, int cache_threshold):
        ModelWithCM<Val>(model_id, num_params, cache_threshold),
        frequency_(kvstore::RangeManager::Get().GetChunkNum(model_id), 0) {}

   protected:
    void replace_lock(int num_to_replace, std::vector<size_t>& chunks_to_replace) override {
        std::vector<std::pair<size_t, int>> pool;
        pool.reserve(num_cached_);
        while (pool.size() < num_to_replace) {
            for (size_t i = 0; i < num_chunks_; ++i) {
                if (is_cached_[i] && prepare_count_[i] == 0 && mtx_[i].try_lock()) {  // get unprotected unlocked chunks
                    if (is_cached_[i] && prepare_count_[i] == 0) {
                        pool.push_back(std::make_pair(i, frequency_[i]));
                    } else {
                        mtx_[i].unlock();
                    }
                }
            }
            if (pool.size() < num_to_replace) {
                std::this_thread::yield();
            }
        }

        // sort according to frequency in ascending order
        std::sort(pool.begin(), pool.end(), [](std::pair<size_t, int> a, std::pair<size_t, int> b) {
            return a.second < b.second;
        });

        // get only the least frequent chunks
        chunks_to_replace.resize(num_to_replace);
        for (int i = 0; i < num_to_replace; ++i) {
            chunks_to_replace[i] = pool[i].first;
            frequency_[pool[i].first] = 0;
        }
        // Unlock the other chunks
        if (pool.size() > num_to_replace) {
            for (int i = num_to_replace; i < pool.size(); ++i) {
                mtx_[pool[i].first].unlock();
            }
        }
    }

    inline void touch(size_t chunk_id) override { frequency_[chunk_id] += 1; }

    std::vector<int> frequency_;
};

template<typename Val>
class ModelWithCMRandom : public ModelWithCM<Val> {
   public:
    using ChunkBasedModel<Val>::num_chunks_;
    using ChunkBasedModel<Val>::is_cached_;
    using ModelWithCM<Val>::num_cached_;
    using ModelWithCM<Val>::prepare_count_;
    using ChunkBasedMTModel<Val>::mtx_;

    ModelWithCMRandom(int model_id, int num_params, int cache_threshold):
        ModelWithCM<Val>(model_id, num_params, cache_threshold) {}

   protected:
    void replace_lock(int num_to_replace, std::vector<size_t>& chunks_to_replace) override {
        // 1. Get chunks that can be flushed
        chunks_to_replace.reserve(num_cached_);
        while (chunks_to_replace.size() < num_to_replace) {
            for (size_t i = 0; i < num_chunks_; ++i) {
                if (is_cached_[i] && prepare_count_[i] == 0 && mtx_[i].try_lock()) {
                    if (is_cached_[i] && prepare_count_[i] == 0) {
                        chunks_to_replace.push_back(i);
                        if (chunks_to_replace.size() == num_to_replace) break;
                    } else {
                        mtx_[i].unlock();
                    }
                }
            }
            if (chunks_to_replace.size() < num_to_replace) {
                std::this_thread::yield();
            }
        }
    }
};

}  // namespace model
}  // namespace ml
