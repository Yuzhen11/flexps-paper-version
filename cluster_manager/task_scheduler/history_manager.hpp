#pragma once

#include <unordered_map>
#include <vector>
#include <memory>

#include "core/instance.hpp"

namespace husky {

class HistoryManager {
public:
    static HistoryManager& get() {
        static HistoryManager history_manager;
        return history_manager; 
    }

    // init some data
    void start(int num_processes) {
        num_processes_ = num_processes;
    }

    void update_history(int task_id, const std::vector<std::pair<int, int>>& pid_tids) {
        // if the task with this task_id runs first time, init its history first
        auto it = history_.find(task_id);
        if (it == history_.end()) {
            history_.emplace(task_id, std::vector<int>(num_processes_));
        }

        // remove repeated
        std::vector<int> pids;
        for(int i = 0; i < pid_tids.size(); i++) {
            pids.push_back(pid_tids[i].first);
        }

        std::sort(pids.begin(), pids.end());
        pids.erase(std::unique(pids.begin(), pids.end()), pids.end());

        // update history
        for (auto pid : pids) {
            history_.at(task_id)[pid] += 1;
        }
    }

    const std::vector<int>& get_task_history(int task_id) {
        return history_[task_id];
    }

    void set_last_instance(int task_id, const std::shared_ptr<Instance>& last_instance) {
        last_instance_[task_id] = last_instance;
    }

    const std::shared_ptr<Instance>& get_last_instance(int task_id) {
        return last_instance_[task_id];
    }
private:
    std::unordered_map<int, std::vector<int>> history_;
    std::unordered_map<int, std::shared_ptr<Instance>> last_instance_;
    // the num of processes in the cluster,it is also the history_ vector size
    int num_processes_;
};

}
