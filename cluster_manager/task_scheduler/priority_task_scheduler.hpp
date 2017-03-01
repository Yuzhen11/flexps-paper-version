#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include "core/constants.hpp"
#include "cluster_manager/task_scheduler/available_workers.hpp"
#include "cluster_manager/task_scheduler/task_manager.hpp"
#include "cluster_manager/task_scheduler/task_scheduler.hpp"

#include "core/color.hpp"

namespace husky {

/*
 * PriorityTaskScheduler implements TaskScheduler
 *
 * It can disitribute more than one task at each time
 */
class PriorityTaskScheduler : public TaskScheduler {

   public:
    PriorityTaskScheduler(WorkerInfo& worker_info_) : TaskScheduler(worker_info_) {
        num_processes_ = worker_info_.get_num_processes();
        auto tids = worker_info_.get_global_tids();
        for (auto tid : tids) {
            available_workers_.add_worker(worker_info_.get_process_id(tid), tid);
        }
    }

    ~PriorityTaskScheduler() override {}

    void init_tasks(const std::vector<std::shared_ptr<Task>>& tasks) override;

    // 1. add this thread to available_worker
    // 2. untrack this thread for that instance
    void finish_thread(int instance_id, int global_thread_id) override;

    // 1. assign available threads to ready tasks
    // 2. produce actual instances
    std::vector<std::shared_ptr<Instance>> extract_instances() override;

    // check whether all the tasks have finished all epochs
    bool is_finished() override;

   private:
    // maintain available workers 
    AvailableWorkers available_workers_;

    // maintain the status of tasks and help find their preference
    TaskManager task_manager_;

    // number of processes in the cluster
    int num_processes_;
};

}  // namespace husky
