#pragma once

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <utility>
#include <stdio.h>
#include <vector>

#include "core/constants.hpp"
#include "ml/model/chunk_based_mt_model.hpp"
#include "kvstore/kvstore.hpp"

namespace ml {
namespace model {

enum CacheStatus { InKVStore = 0, InDisk = 1, CachedNew = 2, CachedOld = 3 };

class ChunkFileEditor {
   public:
    ChunkFileEditor(std::vector<std::vector<float>>* params, int chunk_size, int last_chunk_size, int num_chunks):
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
        for (auto id : ids) {
            auto iter = chunk_map_.find(id);
            if (iter == chunk_map_.end()) throw husky::base::HuskyException("Target chunk is not in disk");

            // allocate memory to chunk
            int size = chunk_size_;
            if (id == num_chunks_ - 1) size = last_chunk_size_;
            chunks_ptr_->at(id).resize(size);

            fseek(file_, iter->second, SEEK_SET);
            fread(chunks_ptr_->at(id).data(), sizeof(float), size, file_);
        }
    }

    void write_chunks(const std::vector<size_t>& ids) {
        for (auto id : ids) {
            auto iter = chunk_map_.find(id);
            // get actual size of chunk
            int size = chunk_size_;
            if (id == num_chunks_ - 1) size = last_chunk_size_;

            if (iter != chunk_map_.end()) {  // if the chunk is previously written
                fseek(file_, iter->second, SEEK_SET);
                fwrite(chunks_ptr_->at(id).data(), sizeof(float), size, file_);
            } else {  // first time to write the chunk
                chunk_map_[id] = bytes_written_;
                fseek(file_, 0, SEEK_END);
                fwrite(chunks_ptr_->at(id).data(), sizeof(float), size, file_);
                bytes_written_ += size * sizeof(float);
            }
        }
    }

   protected:
    FILE* file_;
    std::unordered_map<size_t, int> chunk_map_;  // chunk_id, byte position
    std::vector<std::vector<float>> * chunks_ptr_ = NULL;
    int bytes_written_ = 0;
    int last_chunk_size_;
    int chunk_size_;
    int num_chunks_;
};

class ModelWithCM : public ChunkBasedMTLockModel {
   public:
    ModelWithCM(int model_id, int num_params, int cache_threshold = 0) :
        ChunkBasedMTLockModel(model_id, num_params), cache_threshold_(cache_threshold),
        status_(num_chunks_, 0),
        cfe_(&params_, kvstore::RangeManager::Get().GetChunkSize(model_id), kvstore::RangeManager::Get().GetLastChunkSize(model_id), num_chunks_) {}

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
        ChunkBasedModel::Push(keys, vals);

        {
            // 3. Unlock the chunks
            std::lock_guard<std::mutex> table_lock(chunk_lock_.lock_table_mtx_);
            touch(chunks);
            for (auto chunk_id : chunks) {
                status_[chunk_id] = CachedOld;
            }
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
            auto loc = range_manager.GetLocation(model_id_, keys[i]);
            (*vals)[i] = params_[loc.first][loc.second];
        }

        {
            // 3. Unlock the chunks
            std::unique_lock<std::mutex> table_lock(chunk_lock_.lock_table_mtx_);
            chunk_lock_.unlock_read(chunks);
        }
    }

   protected:
    virtual void Prepare(const std::vector<husky::constants::Key>& keys, int local_id) override {
        std::vector<size_t> chunks_to_fetch;
        std::vector<size_t> chunks_to_prepare;
        auto& range_manager = kvstore::RangeManager::Get();

        std::vector<size_t> chunks_to_replace;
        int num_cached_new = 0;

        {
            // 1. Lock
            std::unique_lock<std::mutex> table_lock(chunk_lock_.lock_table_mtx_);

            // 1. Collect the uncached chunks
            for (size_t i = 0; i < keys.size(); ++i) {
                auto loc = range_manager.GetLocation(model_id_, keys[i]);
                if (status_[loc.first] < 2 && (chunks_to_fetch.empty() || loc.first != chunks_to_fetch.back())) {
                    chunks_to_fetch.push_back(loc.first);
                }
                if (chunks_to_prepare.empty() || loc.first != chunks_to_prepare.back()) {
                    chunks_to_prepare.push_back(loc.first);
                }
            }

            if (chunks_to_fetch.empty()) return;  // Quit prepare if none is missing

            // 2. Try to get the write locks
            bool w_access = chunk_lock_.try_lock_write(chunks_to_fetch);
            while (!w_access) {
                chunk_lock_.cv_.wait(table_lock);

                chunks_to_fetch.clear();
                chunks_to_fetch.reserve(chunks_to_prepare.size());
                for (auto id : chunks_to_prepare) {
                    if (status_[id] < 2) chunks_to_fetch.push_back(id);
                }
                if (chunks_to_fetch.empty()) return;  // no missing chunks now
                w_access = chunk_lock_.try_lock_write(chunks_to_fetch);
            }

            for (auto id : chunks_to_prepare) {
                if (status_[id] >= 2) {
                    status_[id] = CachedNew;  // prevent being replaced
                }
            }

            // 3. Check whether overflow, get the chunks to replace
            num_cached_new = num_cached_ + chunks_to_fetch.size();
            int overflow = num_cached_new - cache_threshold_;
            if (cache_threshold_ != 0 && overflow > 0) {
                num_cached_new = cache_threshold_;
                replace_lock(overflow, chunks_to_replace);
                for (auto chunk_id : chunks_to_replace) {
                    status_[chunk_id] = InDisk;
                    is_cached_[chunk_id] = false;
                }
            }
        }

        // 5. Manipulate different chunks concurrently
        flush_to_disk(chunks_to_replace);
        auto ts = fetch_chunk(chunks_to_fetch, local_id);
        if (ts != -1) wait(ts, local_id);

        {
            // 6. Release the write locks
            std::lock_guard<std::mutex> table_lock(chunk_lock_.lock_table_mtx_);
            for (auto chunk_id : chunks_to_fetch) {
                status_[chunk_id] = CachedNew;
                is_cached_[chunk_id] = true;
            }
            chunk_lock_.unlock_write(chunks_to_fetch);
            chunk_lock_.unlock_write(chunks_to_replace);
            num_cached_ = num_cached_new;
            num_in_disk += chunks_to_replace.size();
        }
    }

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
        std::vector<std::vector<float>*> chunk_ptrs;
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
        num_in_disk -= chunks_disk.size();
        read_from_disk(std::move(chunks_disk));

        return ts;
    }

    virtual void replace_lock(int num_to_replace, std::vector<size_t>& chunks_to_replace) {}
    virtual void touch(const std::vector<size_t>& chunks) {}

    ChunkFileEditor cfe_;
    std::vector<int> status_;
    int num_cached_ = 0;
    int num_in_disk = 0;
    int cache_threshold_;
};

class ModelWithCMLRU : public ModelWithCM {
   public:
    ModelWithCMLRU(int model_id, int num_params, int cache_threshold):
        ModelWithCM(model_id, num_params, cache_threshold),
        recency_(num_chunks_, 0) {}

   protected:
    void replace_lock(int num_to_replace, std::vector<size_t>& chunks_to_replace) override {
        std::vector<std::pair<size_t, int>> pool;
        pool.reserve(num_cached_);
        for (size_t i = 0; i < status_.size(); ++i) {
            if (status_[i] == CachedOld && chunk_lock_.lock_table_[i] == 0) {  // get old unlocked chunks
                pool.push_back(std::make_pair(i, recency_[i]));
            }
        }

        // TODO: strict threshold, wait until pool size is large enough?
        assert(num_to_replace <= pool.size());

        // Sort according to recency in ascending order
        std::sort(pool.begin(), pool.end(), [](std::pair<size_t, int> a, std::pair<size_t, int> b) {
            return a.second < b.second;
        });

        // Get only the least recent chunks
        chunks_to_replace.resize(num_to_replace);
        for (int i = 0; i < num_to_replace; ++i) {
            chunks_to_replace[i] = pool[i].first;
        }

        assert(chunk_lock_.try_lock_write(chunks_to_replace));
    }

    void touch(const std::vector<size_t>& chunks) override {
        ++access_count_;
        for (auto chunk_id : chunks) recency_[chunk_id] = access_count_;
    }

    std::vector<int> recency_;
    int access_count_ = 0;
};

class ModelWithCMLFU : public ModelWithCM {
   public:
    ModelWithCMLFU(int model_id, int num_params, int cache_threshold):
        ModelWithCM(model_id, num_params, cache_threshold),
        frequency_(kvstore::RangeManager::Get().GetChunkNum(model_id), 0) {}

   protected:
    void replace_lock(int num_to_replace, std::vector<size_t>& chunks_to_replace) override {
        std::vector<std::pair<size_t, int>> pool;
        pool.reserve(num_cached_);
        for (size_t i = 0; i < status_.size(); ++i) {
            if (status_[i] == CachedOld && chunk_lock_.lock_table_[i] == 0) {  // get old unlocked chunks
                pool.push_back(std::make_pair(i, frequency_[i]));
            }
        }

        assert(num_to_replace <= pool.size());
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

        assert(chunk_lock_.try_lock_write(chunks_to_replace));
    }


    void touch(const std::vector<size_t>& chunks) override {
        for (auto chunk_id : chunks) frequency_[chunk_id] += 1;
    }

    std::vector<int> frequency_;
};

class ModelWithCMRandom : public ModelWithCM {
   public:
    ModelWithCMRandom(int model_id, int num_params, int cache_threshold):
        ModelWithCM(model_id, num_params, cache_threshold) {}

   protected:
    void replace_lock(int num_to_replace, std::vector<size_t>& chunks_to_replace) override {
        // 1. Get chunks that can be flushed
        std::vector<size_t> pool;
        pool.reserve(num_cached_);
        for (size_t i = 0; i < status_.size(); ++i) {
            if (status_[i] == CachedOld && chunk_lock_.lock_table_[i] == 0) {  // get old unlocked chunks
                pool.push_back(i);
            }
        }

        // 2. Randomly select chunks
        assert(num_to_replace <= pool.size());
        int idx = rand() % pool.size();
        chunks_to_replace.resize(num_to_replace);
        for (int i = 0; i < num_to_replace; ++i) {
            chunks_to_replace[i] = pool[idx];
            ++idx;
            idx %= pool.size();
        }

        // 3. Get write locks
        assert(chunk_lock_.try_lock_write(chunks_to_replace));
    }

    void touch(const std::vector<size_t>& chunks) override {}
};

/*

class ChunkQueue {
   public:
    using ChunkList = std::list<size_t>;
    using ChunkMap = std::unordered_map<size_t, ChunkList::iterator>;

    ChunkQueue() = default;

    std::vector<size_t> pop_back(int num) {
        std::vector<size_t> poped_chunks(num);
        // 1. Update the map
        for (int i = 0; i < num; ++i) {
            chunk_map.erase(chunk_list.back());
            poped_chunks[i] = chunk_list.back();
            chunk_list.pop_back();
        }
        // 3. Return the poped ids
        return poped_chunks;
    }
    
    void to_front(std::vector<size_t>& ids) {
        for (auto id : ids) {
            // 1. erase from the list
            chunk_list.erase(chunk_map[id]);
            // 2. append to the front
            chunk_list.push_front(id);
            // 3. update map
            chunk_map[id] = chunk_list.begin();
        }
    }

    void erase(size_t id) {
        chunk_list.erase(chunk_map[id]);
        chunk_map.erase(id);
    }

    bool is_contained(size_t id) {
        return chunk_map.find(id) != chunk_map.end();
    }

    void push_front(size_t id) {
        chunk_list.push_front(id);
        chunk_map[id] = chunk_list.begin();
    }

    size_t size() {
        return chunk_map.size();
    }

    ChunkList chunk_list;
    ChunkMap chunk_map;
};

class ModelWithCMAdaptive : public ModelWithCM {
   public:
    ModelWithCMAdaptive(int model_id, int num_params, int cache_threshold):
        ModelWithCM(model_id, num_params, cache_threshold),
        frequency_(num_chunks_, 0),
        recency_(num_chunks_, 0) {}

   protected:
    void replace_lock(int num_to_replace, std::vector<size_t>& chunks_to_replace) override {
        std::vector<std::pair<size_t, int>> pool1;
        std::vector<std::pair<size_t, int>> pool2;
        pool1.reserve(num_cached_);
        pool2.reserve(num_cached_);
        for (size_t i = 0; i < status_.size(); ++i) {
            if (status_[i] == CachedOld && chunk_lock_.lock_table_[i] == 0) {  // get old unlocked chunks
                if (frequency_[i] == 1) {
                    pool1.push_back(std::make_pair(i, recency_[i]));
                } else {
                    pool2.push_back(std::make_pair(i, recency_[i]));
                }
            }
        }

        // sort according to recency in ascending order
        std::sort(pool1.begin(), pool1.end(), [](std::pair<size_t, int> a, std::pair<size_t, int> b) {
            return a.second < b.second;
        });
        std::sort(pool2.begin(), pool2.end(), [](std::pair<size_t, int> a, std::pair<size_t, int> b) {
            return a.second < b.second;
        });

        int first_time_count = num_cached_ - frequent_count_;
            // get only the least recent chunks
        int max_evict = first_time_count - target_;
        int num_evict2 = 0;
        int num_evict1 = 0;
        if (first_time_count > target_ || first_time_count == cache_threshold_) {
            if (max_evict > num_to_replace) {
                num_evict1 = std::min(static_cast<size_t>(num_to_replace), pool1.size());
                num_evict2 = num_to_replace - num_evict1;
            } else {
                num_evict1 = std::min(static_cast<size_t>(max_evict), pool1.size());
                num_evict2 = num_to_replace - num_evict1;
            }
        } else {
            num_evict2 = std::min(static_cast<size_t>(num_to_replace), pool2.size());
            num_evict1 = num_to_replace - num_evict2;
        }

        chunks_to_replace.reserve(num_to_replace);
        for (int i = 0; i < num_evict1; ++i) {
            chunks_to_replace.push_back(pool1[i].first);
            ghost1.push_front(pool1[i].first);
            frequency_[pool1[i].first] = 0;
        }
        for (int i = 0; i < num_to_replace; ++i) {
            chunks_to_replace.push_back(pool2[i].first);
            ghost2.push_front(pool1[i].first);
            frequency_[pool2[i].first] = 0;
        }
        frequent_count_ -= num_evict2;

        assert(chunk_lock_.try_lock_write(chunks_to_replace));
    }

    void touch(const std::vector<size_t>& chunks) override {
        ++access_count_;
        for (auto chunk_id : chunks) {
            recency_[chunk_id] = access_count_;
            frequency_[chunk_id] += 1;
            if (frequency_[chunk_id] == 2) frequent_count_ += 1;
        }
    }

    void report_miss(const std::vector<size_t>& chunks) override {
        int count1 = 0, count2 = 0, count = 0;
        size_t size1 = ghost1.size();
        size_t size2 = ghost2.size();

        // 1. Remove from ghost 1, 2
        for (auto chunk_id : chunks) {
            if (ghost1.is_contained(chunk_id)) {
                ++count1;
                ghost1.erase(chunk_id);
            } else if (ghost2.is_contained(chunk_id)) {
                ++count2;
                ghost2.erase(chunk_id);
            } else ++count;
        }

        // 2. Update target
        if (size1 >= size2) {
            target_ += count1 - count2 * size1 / size2;
        } else {
            target_ += count1 * size2 / size1 - count2;
        }
        if (target_ < 0) target_ = 0;
        else if (target_ > cache_threshold_) target_ = cache_threshold_;

        // 3. If missing chunk is not in ghost 1 or 2
        if ((num_cached_ - frequent_count_ + ghost1.size()) < cache_threshold_) {
            if (num_cached_ + ghost1.size() + ghost2.size() ==  2 * cache_threshold_) {
                ghost2.pop_back(2);
            }
        } else if (ghost1.size() > 0) { ghost1.pop_back(1); }
    }

    int target_ = 0;  // target first time cache num
    int access_count_ = 0;
    int frequent_count_ = 0;
    std::vector<int> frequency_;
    std::vector<int> recency_;
    ChunkQueue ghost1;
    ChunkQueue ghost2;
};

*/

}  // namespace model
}  // namespace ml
