#pragma once

#include "core/info.hpp"
#include "core/instance.hpp"
#include "core/hash_ring.hpp"
#include "core/worker_info.hpp"

// This file contains function to handle relationships among Info, Instance, HashRing, WorkerInfo...
namespace husky {
namespace utility {

Info instance_to_info(const Instance& instance);

}  // namespace utility
}  // namespace husky
