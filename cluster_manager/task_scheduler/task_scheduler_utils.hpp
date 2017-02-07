#pragma once

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
    if ((instance->get_type() == Task::Type::TwoPhasesTaskType && instance->get_epoch() % 2 == 0) ||
        instance->get_type() == Task::Type::FixedWorkersTaskType) {
        int thread_per_worker = instance->get_num_workers();
        pid_tids = available_workers.get_workers_per_process(thread_per_worker, num_processes);
    } else if (instance->get_type() == Task::Type::TwoPhasesTaskType && instance->get_epoch() % 2 == 1) {
        // run even epoch with 1 thread default
        pid_tids = available_workers.get_workers(1);
    } else if (instance->get_type() == Task::Type::MLTaskType) {
        std::string hint = static_cast<MLTask*>(instance->get_task())->get_hint();
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
