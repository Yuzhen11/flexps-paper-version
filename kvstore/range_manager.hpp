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
    static RangeManager& Get() {
        static RangeManager range_manager;
        return range_manager;
    }

    /*
     * kvstore use this function to set max keys
     */
    void SetMaxKey(int kv_id, husky::constants::Key max_key, int num_servers) { 
        if (kv_id >= server_key_ranges_.size())
            server_key_ranges_.resize(kv_id + 1);

        server_key_ranges_[kv_id].clear();
        for (int i = 0; i < num_servers - 1; ++i) {
            server_key_ranges_[kv_id].push_back(
                pslite::Range(max_key / num_servers * i, max_key / num_servers * (i + 1)));
        }
        // the last range should contain all
        server_key_ranges_[kv_id].push_back(
            pslite::Range(max_key / num_servers * (num_servers - 1), max_key));
    }

    /*
     * kvworker use this function to get the server key ranges
     */
    const std::vector<pslite::Range>& GetServerKeyRanges(int kv_id) {
        assert(kv_id < server_key_ranges_.size());
        return server_key_ranges_.at(kv_id);
    }

   private:
    std::vector<std::vector<pslite::Range>> server_key_ranges_;
};

}  // namespace kvstore
