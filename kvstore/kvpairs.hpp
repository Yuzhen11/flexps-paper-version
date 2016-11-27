#pragma once

namespace kvstore {

/* 
 * Use std::vector first, may replaced by SArray
 */
template<typename Val>
struct KVPairs {
    std::vector<int> keys;
    std::vector<Val> vals;
};

struct PSInfo {
    int channel_id;
    int global_id;
    int num_global_threads;
    int num_ps_servers;
    std::unordered_map<int,int> cluster_id_to_global_id;  // {cluster_id, global_id}

    int get_tid(int cluster_id) {
        return cluster_id_to_global_id[cluster_id];
    }
};

}  // namespace kvstore
