#include "cluster_manager/task_scheduler/history_manager.hpp"

#include <algorithm>
#include <vector>
#include <memory>
#include <iostream>

#include "core/instance.hpp"

namespace husky {

void HistoryManager::start(int num_processes) {
    num_processes_ = num_processes;
}

void HistoryManager::update_history(int task_id, const std::vector<std::pair<int, int>>& pid_tids) {
    // 1. If the task with this task_id runs first time, init its history first
    if (history_.find(task_id) == history_.end()) {
        history_.emplace(task_id, std::vector<int>(num_processes_));
    }

    // 2. Remove repeated
    std::vector<int> pids;
    for(int i = 0; i < pid_tids.size(); i++) {
        pids.push_back(pid_tids[i].first);
    }

    std::sort(pids.begin(), pids.end());
    pids.erase(std::unique(pids.begin(), pids.end()), pids.end());

    // 3. Update history
    for (auto pid : pids) {
        history_.at(task_id).at(pid) += 1;
    }
}

void HistoryManager::clear_history() {
    history_.clear();
    last_instance_.clear();
    num_processes_ = 0;
}

void HistoryManager::set_last_instance(int task_id, const std::shared_ptr<Instance>& last_instance) {
    last_instance_[task_id] = last_instance;
}

} // namespace husky
