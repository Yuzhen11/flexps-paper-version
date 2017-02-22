#pragma once

#include <iostream>
#include <unordered_set>

#include "cluster_manager/task_scheduler/available_workers.hpp"
#include "cluster_manager/task_scheduler/task_scheduler.hpp"
#include "cluster_manager/task_scheduler/history_manager.hpp"

#include "core/color.hpp"

namespace husky {

/*
 * GreedyTaskScheduler implements TaskScheduler
 *
 * It can disitribute more than one task at each time
 */
class GreedyTaskScheduler : public TaskScheduler {
   public:
    GreedyTaskScheduler(WorkerInfo& worker_info_) : TaskScheduler(worker_info_) {
        num_processes_ = worker_info_.get_num_processes();
        // initialize the available_workers_
        auto tids = worker_info_.get_global_tids();
        for (auto tid : tids) {
            available_workers_.add_worker(worker_info_.get_process_id(tid), tid);
        }
    }

    virtual ~GreedyTaskScheduler() override {}

    virtual void init_tasks(const std::vector<std::shared_ptr<Task>>& tasks) override;

    virtual void finish_thread(int instance_id, int global_thread_id) override;

    virtual std::vector<std::shared_ptr<Instance>> extract_instances() override;

    virtual bool is_finished() override;

   private:
    // for debug
    void print_available_workers();

    void print_task_status();

   private:
    int num_processes_;  // num of machines in cluster
    std::vector<std::shared_ptr<Task>> tasks_;
    std::vector<int> task_status_;  // 0: ready to run, 1: running, 2: done
    AvailableWorkers available_workers_;
    std::unordered_map<int, std::unordered_map<int, std::unordered_set<int>>>
        task_id_pid_tids_;  // task_id : pid : {tid1, tid2...}
};

}  // namespace husky
