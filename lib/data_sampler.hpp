#pragma once

#include <cstdlib>
#include "datastore/datastore.hpp"

namespace husky {

template<typename T>
class DataSampler {
   public:
    DataSampler(datastore::DataStore<T>& datastore) : datastore_(datastore) {}
    void random_start_point() {
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
    int chunk_id_ = 0;
    int local_id_ = -1;
    const datastore::DataStore<T>& datastore_;
};

}  // namespace husky
