#pragma once

#include "datastore/datastore.hpp"
#include "datastore/data_store_wrapper.hpp"

namespace datastore {

/*
 * DataLoadBalance: Select a start point and sample the data one by one according to the fixed thread num
 * Can work on the whole datastore
 *
 * Usage:
 *   for (int i = 0; i < num_iters; ++ i) {
 *     auto& data = data_load_balance.next();
 *   }
 */
template<typename T>
class DataLoadBalance {
   public:
    DataLoadBalance() = delete;
    DataLoadBalance(const datastore::DataStore<T>& datastore, int thread_num, int thread_pos) 
        : datastore_(datastore), thread_num_(thread_num), local_id_(thread_pos - thread_num), thread_pos_(thread_pos) {
        DataStoreWrapper<T> wrapper(datastore_);
        is_empty_ = wrapper.empty();
        if (!is_empty_) {
            chunk_id_ = 0;
            // find a non-empty chunk
            while (datastore_[chunk_id_].empty()) {
                chunk_id_ += 1;
                chunk_id_ %= datastore_.size();
            }
        }
    }
    /*
     * Whether the internal datastore is empty
     */
    bool empty() {
        return is_empty_;
    }

    void reset() {
        local_id_ = thread_pos_ - thread_num_;
        chunk_id_ = 0;
    }

    bool has_next() {
        if (is_empty_)
            return false;
        local_id_ += thread_num_;
        while (true) {
            if (chunk_id_ == datastore_.size() - 1 && local_id_ >= datastore_[datastore_.size()-1].size()) {
                return false;
            }
            if (local_id_ < datastore_[chunk_id_].size()) {
                return true;
            }
            else {
                local_id_ = thread_pos_;
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
    // num of threads
    int thread_num_;
    int thread_pos_;
    const datastore::DataStore<T>& datastore_;
    bool is_empty_ = false;
};

}  // namespace datastore
