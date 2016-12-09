#pragma once

#include "core/info.hpp"
#include "core/instance.hpp"
#include "husky/core/worker_info.hpp"

// This file contains function to handle relationships among Info, Instance, WorkerInfo...
namespace husky {
namespace utility {
namespace {

Info instance_to_info(const Instance& instance, const WorkerInfo& worker_info_, std::pair<int,int> tid_cid) {
    int pid = worker_info_.get_process_id();

    Info info;
    info.set_local_id(tid_cid.first);
    info.set_global_id(worker_info_.local_to_global_id(tid_cid.first));
    info.set_cluster_id(tid_cid.second);

    WorkerInfo worker_info;
    worker_info.set_process_id(pid);
    std::unordered_map<int,int> cluster_id_to_global_id;
    auto& cluster = instance.get_cluster();
    for (auto& kv : cluster) {
        auto proc_id = kv.first;
        for (auto p : kv.second) {
            // Setup cluster_id_to_global_id map
            cluster_id_to_global_id.insert({p.second, p.first});  // {cluster_id, global_id}

            worker_info.add_worker(proc_id, p.first, worker_info_.global_to_local_id(p.first));
        }
    }
    info.set_worker_info(std::move(worker_info));
    info.set_cluster_global(std::move(cluster_id_to_global_id));

    return info;
}

}

}  // namespace utility
}  // namespace husky
