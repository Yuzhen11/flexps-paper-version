#include "cluster_manager/task_scheduler/task_manager.hpp"

#include <algorithm>
#include <list>
#include <unordered_set> 
#include <unordered_map> 

namespace husky {

void TaskManager::add_tasks(const std::vector<std::shared_ptr<Task>>& tasks) {
    for (auto task : tasks) {
        task_priority_.push_back(std::make_pair(task->get_id(), 0));
        tasks_.push_back(task);
    }
    num_tasks_ = tasks_.size(); 
    task_status_.resize(num_tasks_, 0); 
    task_rejected_times_.resize(num_tasks_, 0);
}

bool TaskManager::is_finished() {
    for (auto& task_status : task_status_) {
        if (task_status != 2) {
            return false;
        }
    }
    return true;
}

// try to finish this epoch and mark the task as fininshed or ready
void TaskManager::finish_thread(int task_id, int pid, int global_thread_id) {
    auto& threads = task_id_pid_tids_[task_id][pid];
    threads.erase(global_thread_id);
    if (threads.empty()) {
        task_id_pid_tids_[task_id].erase(pid);
        if (task_id_pid_tids_[task_id].size() == 0) {
            tasks_[task_id]->inc_epoch();
            if (tasks_[task_id]->get_current_epoch() == tasks_[task_id]->get_total_epoch()) {
                task_status_.at(task_id) = 2;  // mark it as finished
            } else {
                task_status_.at(task_id) = 0;  // mark it as ready
            }
        }
    }
}

std::vector<int> TaskManager::order_by_priority() {
    std::sort(task_priority_.begin(), task_priority_.end(), 
            [](const std::pair<int, int>& left, const std::pair<int, int>& right) {
        return left.second > right.second; 
    });

    std::vector<int> sorted_ready_id;
    for (auto& id_pri : task_priority_) {
        if (task_status_.at(id_pri.first) == 0) {
            sorted_ready_id.push_back(id_pri.first);
        } 
    }
    return sorted_ready_id;
}

std::vector<int> TaskManager::get_preferred_proc(int task_id) {
    std::vector<int> task_history = HistoryManager::get().get_task_history(task_id);
    std::vector<int> plan;
    int smallest = 9999;
    int largest = 0;
    if (task_history.size() == 0) {
        return {};// no history then no preferrence let scheduler choose
    } else {
        for (int i=0; i<task_history.size(); i++) {
            if (task_history[i] < smallest) {
                smallest = task_history[i];
            }
            if (task_history[i] > largest) {
                largest = task_history[i];
            }
        }

        for (int i=0; i<task_history.size(); i++) {
            if (task_history[i] == smallest) {
                plan.push_back(i);
            }
        }
    }
    return plan;
}

void TaskManager::suc_sched(int task_id) {
    assert(task_id>=0 && task_id < num_tasks_);
    task_status_[task_id] = 1; // running
    task_rejected_times_[task_id] = 0;
    auto it = std::find_if(task_priority_.begin(), task_priority_.end(), 
            [&task_id](const std::pair<int, int>& element) {return element.first == task_id;} );
    if (it != task_priority_.end()) {
        (*it).second = 0;
    }

    if (in_angry_list(task_id)) {
        erase_angry_task(task_id);
    }
}

void TaskManager::fail_sched(int task_id) {
    assert(task_id>=0 && task_id<num_tasks_);
    task_rejected_times_.at(task_id) += 1;
    auto it = std::find_if(task_priority_.begin(), task_priority_.end(), 
            [&task_id](const std::pair<int, int>& element) {return element.first == task_id;} );
    if (it != task_priority_.end()) {
        (*it).second += 1;
        if ((*it).second >= angry_threshold_) {
            if (!in_angry_list(task_id)) {
                angry_list_.push_back(task_id); // append at the end;
                task_status_[task_id] = 3;
            }
        }
    }
}

void TaskManager::record_and_track(int task_id, std::vector<std::pair<int, int>>& pid_tids) {
    assert(task_id>=0 && task_id<num_tasks_);
    HistoryManager::get().update_history(task_id, pid_tids);
    for (auto& pid_tid : pid_tids) {
        task_id_pid_tids_[task_id][pid_tid.first].insert(pid_tid.second);
    }
}

std::shared_ptr<Task> TaskManager::get_task_by_id(int task_id) {
    assert(task_id>=0 && task_id < num_tasks_);
    return tasks_[task_id];
}

int TaskManager::get_task_priority(int task_id) {
    assert(task_id>=0 && task_id < num_tasks_);
    auto it = std::find_if(task_priority_.begin(), task_priority_.end(), 
            [&task_id](const std::pair<int, int>& element) {return element.first == task_id;} );
    if (it != task_priority_.end()) {
        return (*it).second;
    }
    return -1;
}

int TaskManager::get_task_status(int task_id) {
    assert(task_id>=0 && task_id < num_tasks_);
    return task_status_[task_id];
}

int TaskManager::get_task_rej_times(int task_id) {
    assert(task_id>=0 && task_id < num_tasks_);
    return task_rejected_times_[task_id];
}

int TaskManager::get_num_tasks() {
    return num_tasks_;
}

std::unordered_set<int> TaskManager::get_tracking_threads(int task_id, int proc_id) {
    if (task_id_pid_tids_.find(task_id) != task_id_pid_tids_.end()) {
        if (task_id_pid_tids_[task_id].find(proc_id) != task_id_pid_tids_[task_id].end()) {
            return task_id_pid_tids_[task_id][proc_id];
        }
    }
    return {};
}

bool TaskManager::exist_angry_tasks() {
    return (!angry_list_.empty());
}

bool TaskManager::in_angry_list(int task_id) {
    for (auto ele : angry_list_) {
        if (ele == task_id)
            return true;
    }
    return false;
}

void TaskManager::erase_angry_task(int task_id) {
    angry_list_.remove_if([&task_id](int ele){return ele == task_id; });
}

} // namespace husky
