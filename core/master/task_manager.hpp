#pragma once

#include <unordered_map>
#include "core/common/task.hpp"
#include "base/log.hpp"

namespace husky {

// Use to manage tasks in Master
class TaskManager {
public:
    TaskManager() = default;
    void add_task(const Task& task) {
        task_map.insert({task.get_task_id(), task});
        base::log_msg("[TaskManager]: Task: "+std::to_string(task.get_task_id())+" added");
    }
private:
    std::unordered_map<int, Task> task_map;
};
}  // namespace husky
