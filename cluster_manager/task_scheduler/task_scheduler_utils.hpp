#pragma once

#include <boost/algorithm/string.hpp>
#include <vector>

#include "cluster_manager/task_scheduler/available_workers.hpp"
#include "core/instance.hpp"
#include "core/task.hpp"
#include "core/constants.hpp"

namespace husky {

void instance_basic_setup(std::shared_ptr<Instance>& instance, const Task& task);

/*
 * guarantee each task threads is not greater than all threads in cluster
 */
void global_guarantee_threads(const std::vector<std::shared_ptr<Task>>& tasks); 

/*
 * Using task history information to let task travel in different processes
 */
std::vector<int> get_preferred_proc(int task_id);

std::vector<std::pair<int, int>> select_threads_from_subset(
        std::shared_ptr<Instance>& instance, AvailableWorkers& available_workers, int num_processes, 
        int required_num_threads, const std::vector<int>& candidate_proc);

}  // namespace husky
