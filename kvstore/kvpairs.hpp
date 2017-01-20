#pragma once
#include <vector>
#include "kvstore/ps_lite/sarray.h"
#include "core/constants.hpp"

namespace kvstore {

/*
 * Use std::vector first, may replaced by SArray
 */
template <typename Val>
struct KVPairs {
    pslite::SArray<husky::constants::Key> keys;
    pslite::SArray<Val> vals;
};

struct PSInfo {
    int channel_id;
    int global_id;
    int num_global_threads;
    int num_ps_servers;
    std::unordered_map<int, int> cluster_id_to_global_id;  // {cluster_id, global_id}

    int get_tid(int cluster_id) { return cluster_id_to_global_id[cluster_id]; }
};

}  // namespace kvstore
