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
    /*
     * Whether the internal datastore is empty
     */
    bool empty() {
        return data_sampler_.empty();
    }
    int get_batch_size() const {
        return batch_size_;
    }
    void random_start_point() {
        data_sampler_.random_start_point();
    }
    /*
     * \return keys in next_batch
     * store next batch data pointer in batch_data_, doesn't own the data
     */
    std::vector<husky::constants::Key> prepare_next_batch() {
        if (empty())
            return {};
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
    void prepare_next_batch_data() {
        if (empty())
            return;
        for (int i = 0; i < batch_size_; ++ i) {
            auto& data = data_sampler_.next();
            batch_data_[i] = const_cast<T*>(&data);
        }
    }

    const std::vector<T*>& get_data_ptrs() {
        if (empty())
            return empty_batch_;
        else
            return batch_data_;  // batch_data_ should have been prepared
    }
   private:
    DataSampler<T> data_sampler_;
    int batch_size_;
    std::vector<T*> batch_data_;
    std::vector<T*> empty_batch_;  // Only for empty batch usage
};

}  // namespace datastore
