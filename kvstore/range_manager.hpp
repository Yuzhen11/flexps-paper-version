#pragma once

#include <cassert>
#include <vector>
#include "kvstore/ps_lite/range.h"

namespace kvstore {

/*
 * A singleton to manage the range in kvstore
 *
 * Not thread-safe, all the kvstore should be set before execution
 */
class RangeManager {
   public:
    /*
     * Assume the first guy call the Get will pass the num_servers parameter
     *
     * CreateKVStore will be the first one who calls this function
     */
    static RangeManager& Get(int num_servers = -1) {
        static RangeManager range_manager(num_servers);
        return range_manager;
    }

    /*
     * kvstore use this function to set max keys
     *
     * The default chunk_size is 100
     *
     * The strategy is:
     * 1. divide max_key into chunks, 
     * 2. assign ranges according to chunks
     *
     * Example:
     * max_key = 13, chunk_size = 5, num_servers = 2
     * chunks: {5, 5, 3}
     * servers: {{5}, {5, 3}} -> {{0, 5}, {5, 13}}
     */
    void SetMaxKeyAndChunkSize(int kv_id, 
            husky::constants::Key max_key = std::numeric_limits<husky::constants::Key>::max(),
            int chunk_size = default_chunk_size_) { 
        assert(num_servers_ > 0);
        if (kv_id >= server_key_ranges_.size()) {
            server_key_ranges_.resize(kv_id + 1);
            max_keys_.resize(kv_id+1);
            chunk_sizes_.resize(kv_id+1);
            chunk_nums_.resize(kv_id+1);
        }

        // If there's one set, overwrite it
        max_keys_[kv_id] = max_key;
        server_key_ranges_[kv_id].clear();
        // 1. Set chunk size
        chunk_sizes_[kv_id] = chunk_size;
        chunk_nums_[kv_id] = (max_key-1)/chunk_size + 1;

        // 2. Set server_key_ranges
        int chunk_num = chunk_nums_[kv_id];

        for (int i = 0; i < num_servers_ - 1; ++i) {
            server_key_ranges_[kv_id].push_back(
                pslite::Range(chunk_num / num_servers_ * i * chunk_size, 
                              chunk_num / num_servers_ * (i + 1) * chunk_size));
        }
        // the last range should contain all
        server_key_ranges_[kv_id].push_back(
            pslite::Range(chunk_num / num_servers_ * (num_servers_ - 1) * chunk_size, max_key));
    }

    /*
     * kvworker use this function to get the server key ranges
     */
    const std::vector<pslite::Range>& GetServerKeyRanges(int kv_id) {
        assert(kv_id < server_key_ranges_.size());
        return server_key_ranges_.at(kv_id);
    }

    size_t GetChunkSize(int kv_id) {
        return chunk_sizes_[kv_id];
    }
    size_t GetChunkNum(int kv_id) {
        return chunk_nums_[kv_id];
    }
    size_t GetLastChunkSize(int kv_id) {
        return max_keys_[kv_id]%chunk_sizes_[kv_id];
    }

   private:
    RangeManager(int num_servers) : num_servers_(num_servers) {}

    int num_servers_;

    std::vector<std::vector<pslite::Range>> server_key_ranges_;
    std::vector<husky::constants::Key> max_keys_;
    std::vector<size_t> chunk_sizes_;
    std::vector<size_t> chunk_nums_;

    static const int default_chunk_size_ = 100;
};

}  // namespace kvstore
