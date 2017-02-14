#pragma once

#include "datastore/datastore.hpp"

namespace datastore {

/*
 * DataLoadBalance: Select a random start point and sample the data one by one according to the fixed thread num
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
    DataLoadBalance(datastore::DataStore<T>& datastore, int thread_num, int thread_pos) : datastore_(datastore), thread_num_(thread_num), local_id_(thread_pos - thread_num), thread_pos_(thread_pos) {}
    bool has_next() {
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
    void start_point() {
      chunk_id_ = 0;
      // find a non-empty chunk
      while (datastore_[chunk_id_].empty()) {
          chunk_id_ += 1;
          chunk_id_ %= datastore_.size();
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
};

}  // namespace datastore
