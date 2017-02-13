#pragma once

#include "available_workers.hpp"
#include "task_scheduler.hpp"
#include "task_scheduler_utils.hpp"
#include "history_manager.hpp"

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
        // init history manager map
        HistoryManager::get().start(num_processes_);
        // initialize the available_workers_
        auto tids = worker_info_.get_global_tids();
        for (auto tid : tids) {
            available_workers_.add_worker(worker_info_.get_process_id(tid), tid);
        }
    }

    virtual ~SequentialTaskScheduler() override {}

    virtual void init_tasks(const std::vector<std::shared_ptr<Task>>& tasks) override {
        for (auto& task : tasks) {
            tasks_queue_.push(task);
        }
    }
    virtual void finish_thread(int instance_id, int global_thread_id) override {
        int proc_id = worker_info.get_process_id(global_thread_id);
        auto& threads = task_id_pid_tids_[instance_id][proc_id];
        // remove from task_id_pid_tids_
        threads.erase(global_thread_id);
        available_workers_.add_worker(proc_id, global_thread_id);
        if (threads.empty()) {
            task_id_pid_tids_[instance_id].erase(proc_id);
            if (task_id_pid_tids_[instance_id].size() == 0) {  // If all the processes are done, the instance is done
                task_id_pid_tids_.erase(instance_id);

                auto& task = tasks_queue_.front();
                task->inc_epoch();  // Trying to work on next epoch
                if (task->get_current_epoch() ==
                    task->get_total_epoch()) {  // If all the epochs are done, then task is done
                    tasks_queue_.pop();
                }
            }
        }
    }
    virtual std::vector<std::shared_ptr<Instance>> extract_instances() override {
        // Assign no instance if 1. task_queue is empty or 2. current instance is still running
        if (tasks_queue_.empty() || !task_id_pid_tids_.empty())
            return {};
        auto& task = tasks_queue_.front();
        auto instance = task_to_instance(*task);
        return {instance};
    }
    virtual bool is_finished() override { return tasks_queue_.empty(); }

   private:
    /*
     * Generate real running instance from task
     *
     * Be careful that instance may not be complete within the process:
     * Empty instance -> set_task() -> set_num_workers() -> add_thread() -> complete instance
     */
    std::shared_ptr<Instance> task_to_instance(const Task& task) {
        // create the instance
        std::shared_ptr<Instance> instance(new Instance);
        instance_basic_setup(instance, task);

        // select threads according to the instance
        std::vector<std::pair<int,int>> pid_tids = select_threads(instance, available_workers_, num_processes_);

        // If requirement is satisfied, set the instance
        if (!pid_tids.empty()) {
            int j = 0;
            for (auto pid_tid : pid_tids) {
                instance->add_thread(pid_tid.first, pid_tid.second, j++);
                task_id_pid_tids_[instance->get_id()][pid_tid.first].insert(pid_tid.second);
            }
            // update history
            HistoryManager::get().update_history(instance->get_id(), pid_tids);
        } else {
            throw base::HuskyException("[Sequential Task Scheduler] Cannot assign next instance");
        }
        return instance;
    }

   private:
    std::queue<std::shared_ptr<Task>> tasks_queue_;
    AvailableWorkers available_workers_;
    int num_processes_;  // num of machines in cluster
    std::unordered_map<int, std::unordered_map<int, std::unordered_set<int>>> 
        task_id_pid_tids_;  // task_id : pid : {tid1, tid2...}
};

}  // namespace husky
