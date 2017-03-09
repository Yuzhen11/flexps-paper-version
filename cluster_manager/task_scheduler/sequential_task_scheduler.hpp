#pragma once

#include <memory>
#include <queue>
#include <vector>

#include "cluster_manager/task_scheduler/available_workers.hpp"
#include "cluster_manager/task_scheduler/task_scheduler.hpp"
#include "core/instance.hpp"
#include "core/task.hpp"

namespace husky {

/*
 * SequentialTaskScheduler implements TaskScheduler
 *
 * It basically extracts and executes tasks from queue one by one
 */
class SequentialTaskScheduler : public TaskScheduler {
   public:
    SequentialTaskScheduler(WorkerInfo& worker_info_) : TaskScheduler(worker_info_) {
        num_processes_ = worker_info_.get_num_processes();
        // initialize the available_workers_
        auto tids = worker_info_.get_global_tids();
        for (auto tid : tids) {
            available_workers_.add_worker(worker_info_.get_process_id(tid), tid);
        }
    }

    virtual ~SequentialTaskScheduler() override {}

    virtual void init_tasks(const std::vector<std::shared_ptr<Task>>& tasks) override; 

    virtual void finish_thread(int instance_id, int global_thread_id) override;

    virtual std::vector<std::shared_ptr<Instance>> extract_instances() override;

    virtual bool is_finished() override;

   private:
    /*
     * Using task history information to let task travel in different processes
     */
    std::vector<int> get_preferred_proc(int task_id);

    /*
     * Generate real running instance from task
     *
     * Be careful that instance may not be complete within the process:
     * Empty instance -> set_task() -> set_num_workers() -> add_thread() -> complete instance
     */
    std::shared_ptr<Instance> task_to_instance(const Task& task);

   private:
    std::queue<std::shared_ptr<Task>> tasks_queue_;
    AvailableWorkers available_workers_;
    int num_processes_;  // num of machines in cluster
    std::unordered_map<int, std::unordered_map<int, std::unordered_set<int>>> 
        task_id_pid_tids_;  // task_id : pid : {tid1, tid2...}
};

} // namespace husky
