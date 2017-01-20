#pragma once

#include <boost/algorithm/string.hpp>

#include "core/instance.hpp"
#include "core/task.hpp"

namespace husky {
namespace {

void instance_basic_setup(std::shared_ptr<Instance>& instance, const Task& task) {
    // TODO If the task type is MLTaskType and the running type is unset,
    // need to decide it's real running type now
    if (task.get_type() == Task::Type::MLTaskType) {
        // TODO now set to SingleTaskType for testing...
        instance->set_task(task, "");
        // instance->set_task(task, Task::Type::HogwildTaskType);
    } else {
        instance->set_task(task);
    }

    // TODO: ClusterManager needs to design workers number for MLTaskType if user didn't set it
    if (task.get_type() == Task::Type::MLTaskType && task.get_num_workers() == 0)
        instance->set_num_workers(1);
}

std::vector<std::pair<int, int>> select_threads(std::shared_ptr<Instance>& instance, AvailableWorkers& available_workers, int num_processes) {
    // randomly select threads
    std::vector<std::pair<int, int>> pid_tids;
    if (instance->get_type() == Task::Type::ConfigurableWorkersTaskType) {
        std::vector<int> worker_num = static_cast<const ConfigurableWorkersTask*>(instance->get_task())->get_worker_num();
        std::vector<std::string> worker_num_type = static_cast<const ConfigurableWorkersTask*>(instance->get_task())->get_worker_num_type();
        int current_epoch = instance->get_epoch();
        int current_pos = current_epoch % worker_num.size();

        if (worker_num_type[current_pos] == "threads_per_worker") {
	       pid_tids = available_workers.get_workers_per_process(worker_num[current_pos], num_processes);
        }
        else if (worker_num_type[current_pos] == "threads_per_cluster") {
	       pid_tids = available_workers.get_workers(worker_num[current_pos]);
        }
        else if (worker_num_type[current_pos] == "local_threads") {
	       pid_tids = available_workers.get_local_workers(worker_num[current_pos]);
        } else if (worker_num_type[current_pos] == "threads_traverse_cluster") {
            pid_tids = available_workers.get_traverse_workers(instance->get_id(), worker_num[current_pos], num_processes); 
        } else if (worker_num_type[current_pos].find("threads_on_worker") != std::string::npos){
            std::vector<std::string> split_result;
            boost::split(split_result, worker_num_type[current_pos], boost::is_any_of(":"));
            if (split_result.size() == 2) {
                int pos_worker = std::stoi(split_result[1]);
                pid_tids = available_workers.get_workers_exact_process(worker_num[current_pos], pos_worker, num_processes);
            } else {
                husky::LOG_I << "illegal threads_on_worker of worker_num_type!";
            }
        }
        else {
	       husky::LOG_I << "illegal worker_num_type!";
        }
    } else if (instance->get_type() == Task::Type::MLTaskType) {
        std::string hint = static_cast<const MLTask*>(instance->get_task())->get_hint();
        if (hint == "hogwild" || hint == "SPMT:BSP" || hint == "SPMT:SSP" || hint == "SPMT:ASP") {
            // extract from one process
            pid_tids = available_workers.get_local_workers(instance->get_num_workers());
        } else {
            // extract from global workers
            pid_tids = available_workers.get_workers(instance->get_num_workers());
        }
    }
    else {
        // extract from global workers
        pid_tids = available_workers.get_workers(instance->get_num_workers());
    }

    return pid_tids;
}

}  // namespace anonymous
}  // namespace husky
