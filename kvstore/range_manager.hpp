#pragma once

#include <cassert>
#include <cstdlib>
#include <vector>
#include <limits>

#include "kvstore/ps_lite/range.h"
#include "core/constants.hpp"

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
    static RangeManager& Get() {
        static RangeManager range_manager;
        return range_manager;
    }
    RangeManager(const RangeManager&) = delete;
    RangeManager& operator=(const RangeManager&) = delete;
    RangeManager(RangeManager&&) = delete;
    RangeManager& operator=(RangeManager&&) = delete;

    /*
     * Set the num of servers, the server number should be set before used.
     */ 
    void SetNumServers(int num_servers) {
        num_servers_ = num_servers;
    }

    /*
     * To clear all the ranges information
     */
    void Clear();

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
     * servers: {{5, 5}, {3}} -> {{0, 10}, {10, 13}}
     */
    void SetMaxKeyAndChunkSize(int kv_id, 
            husky::constants::Key max_key = std::numeric_limits<husky::constants::Key>::max(),
            int chunk_size = default_chunk_size_);
    
    /*
     * A complicated function for user to directly register the partitions
     */
    void CustomizeRanges(int kv_id,
            husky::constants::Key max_key,
            int chunk_size,
            int chunk_num,
            const std::vector<pslite::Range>& server_key_ranges,
            const std::vector<pslite::Range>& server_chunk_ranges);

    /*
     * Get the server id for a given key
     */
    int GetServerFromKey(int kv_id, husky::constants::Key key) {
        // TODO: can be done in logn
        int i = 0;
        while (key >= server_key_ranges_[kv_id][i].begin()) {
            i += 1;
            if (i == num_servers_)
                break;
        }
        return i-1;
    }
    /*
     * Get the server id for a given chunk id
     */
    int GetServerFromChunk(int kv_id, size_t chunk_id) {
        return GetServerFromKey(kv_id, chunk_id*GetChunkSize(kv_id));
    }

    /*
     * kvworker use this function to get the server key ranges
     */
    const std::vector<pslite::Range>& GetServerKeyRanges(int kv_id) {
        assert(kv_id < server_key_ranges_.size());
        return server_key_ranges_.at(kv_id);
    }

    const std::vector<pslite::Range>& GetServerChunkRanges(int kv_id) {
        assert(kv_id < server_chunk_ranges_.size());
        return server_chunk_ranges_.at(kv_id);
    }

    size_t GetNumRanges() const {
        return server_key_ranges_.size();
    }
    size_t GetChunkSize(int kv_id) {
        return chunk_sizes_[kv_id];
    }
    size_t GetChunkNum(int kv_id) {
        return chunk_nums_[kv_id];
    }
    husky::constants::Key GetMaxKey(int kv_id) {
        return max_keys_[kv_id];
    }
    size_t GetLastChunkSize(int kv_id) {
        if (max_keys_[kv_id]%chunk_sizes_[kv_id] != 0)
            return max_keys_[kv_id]%chunk_sizes_[kv_id];
        else 
            return chunk_sizes_[kv_id];
    }
    int GetNumServers() {
        return num_servers_;
    }

    std::pair<size_t, husky::constants::Key> GetLocation(int kv_id, const husky::constants::Key& key) {  // default range partition
        assert(key < max_keys_[kv_id]);
        std::pair<size_t, husky::constants::Key> loc;
        loc.first = key / chunk_sizes_[kv_id];
        loc.second = key % chunk_sizes_[kv_id];
        return loc;
    }
    static int GetDefaultChunkSize() {
        return default_chunk_size_;
    }

   private:
    RangeManager() = default;

    int num_servers_ = -1;

    std::vector<std::vector<pslite::Range>> server_key_ranges_;
    std::vector<std::vector<pslite::Range>> server_chunk_ranges_;
    std::vector<husky::constants::Key> max_keys_;
    std::vector<size_t> chunk_sizes_;
    std::vector<size_t> chunk_nums_;

    static const int default_chunk_size_ = 100;
};

}  // namespace kvstore
