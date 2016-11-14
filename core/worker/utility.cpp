#include "utility.hpp"

namespace husky {
namespace utility {
Info instance_to_info(const Instance& instance) {
    Info info;
    auto& cluster = instance.get_cluster();
    for (auto& kv : cluster) {
        auto proc_id = kv.first;
        for (auto p : kv.second) {
            info.cluster_id_to_global_id.insert({p.second, p.first});  // {cluster_id, global_id}
        }
    }
    return info;
}

}  // namespace utility
}  // namespace husky
