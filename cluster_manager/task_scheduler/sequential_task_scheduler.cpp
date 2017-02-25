#include "cluster_manager/task_scheduler/sequential_task_scheduler.hpp"

#include <memory>
#include <queue>
#include <vector>

#include "cluster_manager/task_scheduler/available_workers.hpp"
#include "cluster_manager/task_scheduler/task_scheduler.hpp"
#include "cluster_manager/task_scheduler/task_scheduler_utils.hpp"
#include "cluster_manager/task_scheduler/history_manager.hpp"
namespace husky {

void SequentialTaskScheduler::init_tasks(const std::vector<std::shared_ptr<Task>>& tasks) {
    for (auto& task : tasks) {
        tasks_queue_.push(task);
    }
}

void SequentialTaskScheduler::finish_thread(int instance_id, int global_thread_id) {
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

std::vector<std::shared_ptr<Instance>> SequentialTaskScheduler::extract_instances() {
    // Assign no instance if 1. task_queue is empty or 2. current instance is still running
    if (tasks_queue_.empty() || !task_id_pid_tids_.empty())
        return {};
    auto& task = tasks_queue_.front();
    auto instance = task_to_instance(*task);
    return {instance};
}

bool SequentialTaskScheduler::is_finished() { return tasks_queue_.empty(); }

std::shared_ptr<Instance> SequentialTaskScheduler::task_to_instance(const Task& task) {
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

} // namespace husky
