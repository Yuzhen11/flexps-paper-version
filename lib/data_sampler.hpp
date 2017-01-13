#pragma once

#include <cstdlib>
#include "datastore/datastore.hpp"

namespace husky {

/*
 * DataSampler: Select a random start point and sample the data one by one
 */
template<typename T>
class DataSampler {
   public:
    DataSampler() = delete;
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

/*
 * BatchDataSampler: Sample the data in batch
 */
template<typename T>
class BatchDataSampler {
   public:
    BatchDataSampler() = delete;
    BatchDataSampler(datastore::DataStore<T>& datastore, int batch_size) : batch_size_(batch_size), data_sampler_(datastore), batch_data_(batch_size) {}
    int get_batch_size() const {
        return batch_size_;
    }
    void random_start_point() {
        data_sampler_.random_start_point();
    }
    std::vector<int> prepare_next_batch() {
        std::set<int> index_set;  // may use other data structure to de-duplicate
        for (int i = 0; i < batch_size_; ++ i) {
            auto& data = data_sampler_.next();
            batch_data_[i] = const_cast<T*>(&data);
        }
        for (auto data : batch_data_) {
            for (auto field : data->x) {
                index_set.insert(field.fea);
            }
        }
        return {index_set.begin(), index_set.end()};
    }
    const std::vector<T*>& get_data_ptrs() {
        return batch_data_;
    }
   private:
    DataSampler<T> data_sampler_;
    int batch_size_;
    std::vector<T*> batch_data_;
};

/*
 * DataIterator: Iterate all the data in datastore one by one
 *
 * Usage:
 *   while (data_iterator.has_next()) {
 *       auto& data = data_iterator.next();
 *   }
 */
template<typename T>
class DataIterator {
   public:
    DataIterator() = delete;
    DataIterator(datastore::DataStore<T>& datastore) : datastore_(datastore) {}
    bool has_next() {
        local_id_ += 1;
        while (true) {
            if (chunk_id_ == datastore_.size() - 1 && local_id_ == datastore_[datastore_.size()-1].size()) {
                return false;
            }
            if (local_id_ < datastore_[chunk_id_].size()) {
                return true;
            }
            else {
                local_id_ = 0;
                chunk_id_ += 1;
            }
        }
    }
    const T& next() {
        return datastore_[chunk_id_][local_id_];
    }
   private:
    int chunk_id_ = 0;
    int local_id_ = -1;
    const datastore::DataStore<T>& datastore_;
};

/*
 * DataStoreWrapper: This class wraps DataStore and provides more functionality
 */
template<typename T>
class DataStoreWrapper {
   public:
    DataStoreWrapper() = default;
    DataStoreWrapper(datastore::DataStore<T>& datastore) : datastore_(datastore) {}
    // To retrieve the total data size
    size_t get_data_size() {
        if (data_count_ != -1)
            return data_count_;
        data_count_ = 0;
        for (int i = 0; i < datastore_.size(); ++ i) {
            data_count_ += datastore_[i].size();
        }
        return data_count_;
    }
   private:
    const datastore::DataStore<T>& datastore_;

    size_t data_count_ = -1;  // to store the number of data, store it once calculated
};

}  // namespace husky
