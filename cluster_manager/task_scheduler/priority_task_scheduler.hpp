#pragma once

#include <iostream>
#include <vector>
#include <memory>

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

    void finish_thread(int instance_id, int global_thread_id) override;

    std::vector<std::shared_ptr<Instance>> extract_instances() override;

    bool is_finished() override;

   private:
    AvailableWorkers available_workers_;
    TaskManager task_manager_;
    int num_processes_;
};

}  // namespace husky
