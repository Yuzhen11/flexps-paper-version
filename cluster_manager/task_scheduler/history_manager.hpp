#pragma once

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <memory>
#include <iostream>

#include "core/instance.hpp"

namespace husky {

class HistoryManager {
public:
    static HistoryManager& get() {
        static HistoryManager history_manager;
        return history_manager; 
    }

    // init some data
    void start(int num_processes);

    void update_history(int task_id, const std::vector<std::pair<int, int>>& pid_tids);

    void clear_history();

    void set_last_instance(int task_id, const std::shared_ptr<Instance>& last_instance);

    const std::shared_ptr<Instance>& get_last_instance(int task_id) {
        return last_instance_[task_id];
    }

    inline std::vector<int> get_task_history(int task_id) {
        auto it = history_.find(task_id);
        if (it != history_.end()) {
            return history_[task_id];
        }
        return {};
    }

private:
    std::unordered_map<int, std::vector<int>> history_;
    std::unordered_map<int, std::shared_ptr<Instance>> last_instance_;
    // the num of processes in the cluster,it is also the history_ vector size
    int num_processes_;
};

}
