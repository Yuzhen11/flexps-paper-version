#pragma once

#include "datastore/datastore.hpp"

namespace datastore {

/*
 * DataIterator: Iterate all the data in datastore one by one
 *
 * Can work on the whole datastore
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
    DataIterator(const datastore::DataStore<T>& datastore) : datastore_(datastore) {}
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

}  // namespace datastore
