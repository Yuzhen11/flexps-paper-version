#pragma once

#include <cstdlib>
#include "datastore/datastore.hpp"

namespace datastore {

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

}  // namespace datastore
