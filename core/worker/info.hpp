#pragma once
#include <unordered_map>

#include "base/log.hpp"
#include "core/common/hash_ring.hpp"

namespace husky {

struct Info {
    int local_id;
    int global_id;
    int cluster_id;  // The id within this cluster
    HashRing hash_ring;

    std::unordered_map<int,int> cluster_id_to_global_id;  // {cluster_id, global_id}

    int get_tid(int cluster_id) {
        return cluster_id_to_global_id[cluster_id];
    }
    void show() {
        for (auto& kv : cluster_id_to_global_id) {
            base::log_msg("Info: "+std::to_string(kv.first)+" "+std::to_string(kv.second));
        }
    }
};

}  // namespace husky

