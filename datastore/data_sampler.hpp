#pragma once

#include <cstdlib>
#include "datastore/datastore.hpp"
#include "datastore/data_store_wrapper.hpp"

namespace datastore {

/*
 * DataSampler: Select a random start point and sample the data one by one
 * Can work on the whole datastore
 *
 * Usage:
 *   for (int i = 0; i < num_iters; ++ i) {
 *     auto& data = data_sampler.next();
 *   }
 */
template<typename T>
class DataSampler {
   public:
    DataSampler() = delete;
    DataSampler(const DataStore<T>& datastore) : datastore_(datastore) {
        DataStoreWrapper<T> wrapper(datastore_);
        is_empty_ = wrapper.empty();
    }
    /*
     * Whether the internal datastore is empty
     */
    bool empty() {
        return is_empty_;
    }
    void random_start_point() {
        if (empty())
            return;
        chunk_id_ = rand() % datastore_.size();
        // find a non-empty chunk
        while (datastore_[chunk_id_].empty()) {
            chunk_id_ += 1;
            chunk_id_ %= datastore_.size();
        }
        // find a random pos in that chunk
        local_id_ = rand() % datastore_[chunk_id_].size();
        local_id_ -= 1;
    }
    const T& next() {
        assert(!empty());
        local_id_ += 1;  // forward to next position
        if (local_id_ >= datastore_[chunk_id_].size()) {  // if reach the end of chunk, find next available chunk
            chunk_id_ += 1;
            chunk_id_ %= datastore_.size();
            while (datastore_[chunk_id_].empty()) {
                chunk_id_ += 1;
                chunk_id_ %= datastore_.size();
            }
            local_id_ = 0;
        }
        return datastore_[chunk_id_][local_id_];
    }
   private:
    bool is_empty_ = false;
    int chunk_id_ = 0;
    int local_id_ = -1;
    const DataStore<T>& datastore_;
};

}  // namespace datastore
