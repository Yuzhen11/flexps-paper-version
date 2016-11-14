#include "utility.hpp"

namespace husky {
namespace utility {
Info instance_to_info(const Instance& instance, const WorkerInfo& worker_info) {
    Info info;
    auto& cluster = instance.get_cluster();
    for (auto& kv : cluster) {
        auto proc_id = kv.first;
        for (auto p : kv.second) {
            // Setup cluster_id_to_global_id map
            info.cluster_id_to_global_id.insert({p.second, p.first});  // {cluster_id, global_id}

            // Setup hash_ring
            info.hash_ring.insert(p.first, proc_id);  // tid, pid
        }
    }
    return info;
}

}  // namespace utility
}  // namespace husky
