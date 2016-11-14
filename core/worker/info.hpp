#pragma once

namespace husky {

struct Info {
    int local_id;
    int global_id;
    int cluster_id;  // The id within this cluster
};

}  // namespace husky

