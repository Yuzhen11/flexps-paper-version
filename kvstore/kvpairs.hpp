#pragma once
#include <vector>
#include <set>
#include "core/constants.hpp"
#include "kvstore/ps_lite/sarray.h"

namespace kvstore {

template <typename Val>
struct KVPairs {
    pslite::SArray<husky::constants::Key> keys;
    pslite::SArray<Val> vals;
};

struct PSInfo {
    int channel_id;
    int global_id;
    int num_ps_servers;
    std::set<int> local_server_ids;
    std::unordered_map<int, int> server_id_to_global_id;  // {server_id, global_id}

    int get_tid(int server_id) { return server_id_to_global_id[server_id]; }
};

}  // namespace kvstore
