#pragma once
#include <unordered_map>
#include <memory>

#include "base/log.hpp"
#include "core/task.hpp"
#include "core/hash_ring.hpp"
#include "ml/common/mlworker.hpp"

namespace husky {

struct Info {
    int local_id;
    int global_id;
    int cluster_id;  // The id within this cluster
    int proc_id;
    int num_local_threads;  // cluster num locally
    int num_global_threads;  // cluster num in total
    HashRing hash_ring;
    std::shared_ptr<Task> task;
    std::unique_ptr<ml::common::GenericMLWorker> mlworker;

    std::unordered_map<int,int> cluster_id_to_global_id;  // {cluster_id, global_id}

    int get_tid(int cluster_id) const {
        auto p = cluster_id_to_global_id.find(cluster_id);
        return p->second;
    }
    void show() const {
        for (auto& kv : cluster_id_to_global_id) {
            base::log_msg("Info: "+std::to_string(kv.first)+" "+std::to_string(kv.second));
        }
    }
};

}  // namespace husky

