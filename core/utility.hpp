#pragma once

#include "core/info.hpp"
#include "core/instance.hpp"
#include "husky/core/hash_ring.hpp"
#include "husky/core/worker_info.hpp"

// This file contains function to handle relationships among Info, Instance, HashRing, WorkerInfo...
namespace husky {
namespace utility {
namespace {

Info instance_to_info(const Instance& instance, int pid) {
    Info info;
    auto& cluster = instance.get_cluster();
    for (auto& kv : cluster) {
        auto proc_id = kv.first;
        info.global_pids.push_back(proc_id);  // push proc_id
        for (auto p : kv.second) {
            // Setup cluster_id_to_global_id map
            info.cluster_id_to_global_id.insert({p.second, p.first});  // {cluster_id, global_id}
        }
    }

    for (auto& tid_cid : instance.get_threads(pid)) {
        info.local_tids.push_back(tid_cid.first);
    }
    return info;
}

}

}  // namespace utility
}  // namespace husky
