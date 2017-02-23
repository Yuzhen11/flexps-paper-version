#include "kvstore/range_manager.hpp"

namespace kvstore {

void RangeManager::Clear() {
    num_servers_ = -1;
    server_key_ranges_.clear();
    server_chunk_ranges_.clear();
    max_keys_.clear();
    chunk_sizes_.clear();
    chunk_nums_.clear();
}


void RangeManager::SetMaxKeyAndChunkSize(int kv_id, 
        husky::constants::Key max_key,
        int chunk_size) {
    assert(num_servers_ > 0);
    if (kv_id >= server_key_ranges_.size()) {
        server_key_ranges_.resize(kv_id + 1);
        server_chunk_ranges_.resize(kv_id + 1);
        max_keys_.resize(kv_id+1);
        chunk_sizes_.resize(kv_id+1);
        chunk_nums_.resize(kv_id+1);
    }

    // If there's one set, overwrite it
    max_keys_[kv_id] = max_key;
    server_key_ranges_[kv_id].clear();
    server_chunk_ranges_[kv_id].clear();
    // 1. Set chunk size
    chunk_sizes_[kv_id] = chunk_size;
    chunk_nums_[kv_id] = (max_key-1)/chunk_size + 1;

    // 2. Set server_key_ranges
    size_t chunk_num = chunk_nums_[kv_id];

    //  [0, remain)
    size_t base = chunk_num / num_servers_;
    size_t remain = chunk_num % num_servers_;
    for (size_t i = 0; i < remain; ++ i) {
        server_chunk_ranges_[kv_id].push_back(
            pslite::Range(i * (base + 1),
                          (i + 1) * (base + 1)));
    }
    // [remain, num_servers_-1)
    size_t end = remain * (base + 1);
    for (size_t i = 0; i < num_servers_ - remain - 1; ++ i) {
        server_chunk_ranges_[kv_id].push_back(
            pslite::Range((end + i * base),
                          (end + (i + 1) * base)));
    }
    // num_servers_
    server_chunk_ranges_[kv_id].push_back(
        pslite::Range((end + (num_servers_ - remain - 1) * base), chunk_num));

    for (auto range : server_chunk_ranges_[kv_id]) {
        server_key_ranges_[kv_id].push_back(pslite::Range(range.begin()*chunk_size, range.end()*chunk_size));
    }
    server_key_ranges_[kv_id].back() = pslite::Range(server_key_ranges_[kv_id].back().begin(), max_key);  // the last one may overflow

    // Go from the last one to make some modification for the case:
    // max_key: 3, chunk_size: 10, num_server: 3
    // [0,1), [1,1), [1,1)
    // [0,10), [10,10), [10,10) -> [0,3), [3,3), [3,3)
    auto& last_one = server_key_ranges_[kv_id].back();
    if (last_one.begin() > last_one.end())
        last_one = pslite::Range(max_key, max_key);
    for (int i = server_key_ranges_[kv_id].size() - 2; i >= 0; -- i) {
        auto& this_one = server_key_ranges_[kv_id][i];
        auto& next_one = server_key_ranges_[kv_id][i+1];
        if (this_one.end() > next_one.begin()) {
            this_one = pslite::Range(this_one.begin(), next_one.begin());
            if (this_one.begin() > this_one.end()) {
                this_one = pslite::Range(this_one.end(), this_one.end());
            }
        } else {
            break;
        }
    }
}

void RangeManager::CustomizeRanges(int kv_id,
        husky::constants::Key max_key,
        int chunk_size,
        int chunk_num,
        const std::vector<pslite::Range>& server_key_ranges,
        const std::vector<pslite::Range>& server_chunk_ranges) {
    assert(num_servers_ > 0);
    assert(server_key_ranges.size() == num_servers_);
    assert(server_chunk_ranges.size() == num_servers_);
    if (kv_id >= server_key_ranges_.size()) {
        server_key_ranges_.resize(kv_id + 1);
        server_chunk_ranges_.resize(kv_id + 1);
        max_keys_.resize(kv_id+1);
        chunk_sizes_.resize(kv_id+1);
        chunk_nums_.resize(kv_id+1);
    }
    // Set
    server_key_ranges_[kv_id] = server_key_ranges;
    server_chunk_ranges_[kv_id] = server_chunk_ranges;
    chunk_sizes_[kv_id] = chunk_size;
    chunk_nums_[kv_id] = chunk_num;
    max_keys_[kv_id] = max_key;
}

}  // namespace kvstore
