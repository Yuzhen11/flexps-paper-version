#pragma once

#include <boost/algorithm/string.hpp>
#include <vector>

#include "cluster_manager/task_scheduler/available_workers.hpp"
#include "core/instance.hpp"
#include "core/task.hpp"

namespace husky {

void instance_basic_setup(std::shared_ptr<Instance>& instance, const Task& task);

std::vector<std::pair<int, int>> select_threads(std::shared_ptr<Instance>& instance,
        AvailableWorkers& available_workers, int num_processes);

std::vector<std::pair<int, int>> select_threads_from_subset(
        std::shared_ptr<Instance>& instance, AvailableWorkers& available_workers, int num_processes, 
        int required_num_threads, const std::vector<int>& candidate_proc);

}  // namespace husky
