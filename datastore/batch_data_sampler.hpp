#pragma once

#include <set>

#include "core/constants.hpp"
#include "datastore/datastore.hpp"
#include "datastore/data_sampler.hpp"

namespace datastore {

/*
 * BatchDataSampler: Sample the data in batch
 * Can work on the whole datastore
 *
 * Usage:
 *   batch_data_sampler.prepare_next_batch();
 *   for (auto data : get_data_ptrs) {
 *      ...
 *   }
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
    std::vector<husky::constants::Key> prepare_next_batch() {
        std::set<husky::constants::Key> index_set;  // may use other data structure to de-duplicate
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

}  // namespace datastore
