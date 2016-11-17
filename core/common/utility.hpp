#pragma once

#include "core/common/info.hpp"
#include "core/common/instance.hpp"
#include "core/common/hash_ring.hpp"
#include "core/common/worker_info.hpp"

// This file contains function to handle relationships among Info, Instance, HashRing, WorkerInfo...
namespace husky {
namespace utility {

Info instance_to_info(const Instance& instance);

}  // namespace utility
}  // namespace husky
