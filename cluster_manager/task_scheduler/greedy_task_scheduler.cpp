#include "cluster_manager/task_scheduler/greedy_task_scheduler.hpp"

#include <iostream>
#include <unordered_set>

#include "cluster_manager/task_scheduler/available_workers.hpp"
#include "cluster_manager/task_scheduler/task_scheduler.hpp"
#include "cluster_manager/task_scheduler/task_scheduler_utils.hpp"
#include "cluster_manager/task_scheduler/history_manager.hpp"

#include "core/color.hpp"

namespace husky {

void GreedyTaskScheduler::init_tasks(const std::vector<std::shared_ptr<Task>>& tasks) {
    tasks_ = tasks;
    task_status_.clear();
    task_status_.resize(tasks_.size(), 0);
}

void GreedyTaskScheduler::finish_thread(int instance_id, int global_thread_id) {
    int proc_id = worker_info.get_process_id(global_thread_id);
    auto& threads = task_id_pid_tids_[instance_id][proc_id];
    threads.erase(global_thread_id);
    available_workers_.add_worker(proc_id, global_thread_id);
    if (threads.empty()) {  // a process is done
        task_id_pid_tids_[instance_id].erase(proc_id);
        if (task_id_pid_tids_[instance_id].size() == 0) {  // All the processes are done
            task_id_pid_tids_.erase(instance_id);

            // linear search to find the task_id
            auto p = std::find_if(tasks_.begin(), tasks_.end(), [instance_id](const std::shared_ptr<Task>& task) {
                return task->get_id() == instance_id;
            });
            auto& task = *p;
            task->inc_epoch();
            int idx = distance(tasks_.begin(), p);
            if (task->get_current_epoch() == task->get_total_epoch()) {
                task_status_.at(idx) = 2;  // mark it as finished
            } else {
                task_status_.at(idx) = 0;  // mark it as ready
            }
        }
    }
}

std::vector<std::shared_ptr<Instance>> GreedyTaskScheduler::extract_instances() {
    std::vector<std::shared_ptr<Instance>> instances;
    // Go through the tasks list once and assign as much as possible
    for (int i = 0; i < tasks_.size(); ++i) {
        if (task_status_.at(i) == 0) {  // ready to run
            // create the instance
            std::shared_ptr<Instance> instance(new Instance);
            instance_basic_setup(instance, *tasks_[i]);

            int required_num_threads = instance->get_num_workers();
            std::vector<int> candidate_proc = get_preferred_proc(instance->get_id());

            // select threads according to the instance
            std::vector<std::pair<int,int>> pid_tids = select_threads_from_subset(instance, available_workers_, num_processes_, required_num_threads, candidate_proc);

            // If requirement is satisfied, add the instance
            if (!pid_tids.empty()) {
                int j = 0;
                for (auto pid_tid : pid_tids) {
                    instance->add_thread(pid_tid.first, pid_tid.second, j++);
                    task_id_pid_tids_[instance->get_id()][pid_tid.first].insert(pid_tid.second);
                }
                // update history
                HistoryManager::get().update_history(instance->get_id(), pid_tids);
                
                instances.push_back(std::move(instance));
                task_status_.at(i) = 1;
            }
        }
    }
    return instances;
}

bool GreedyTaskScheduler::is_finished() {
    for (auto status : task_status_) {
        if (status != 2)
            return false;
    }
    return true;
}

void GreedyTaskScheduler::print_available_workers() {
    std::cout << "available_workers_: " << available_workers_.get_num_available_workers() << std::endl;
}
void GreedyTaskScheduler::print_task_status() {
    std::cout << "task_scheduler: " << std::endl;
    for (auto status : task_status_) {
        std::cout << status << " ";
    }
    std::cout << std::endl;
}

}  // namespace husky
