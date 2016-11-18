#pragma once

#include <cassert>
#include <functional>
#include <unordered_map>
#include <type_traits>
#include "core/common/task.hpp"
#include "core/common/info.hpp"

namespace husky {

class TaskStore {
public:
    TaskStore() = default;
    /*
     * Add a task into the task_map, the task added should be derived from Task
     */
    template<typename TaskType>
    void add_task(const TaskType& task, const std::function<void(Info)>& func) {
        static_assert(std::is_base_of<Task, TaskType>::value, "TaskType should derived from Task");
        assert(task_map.find(task.get_id()) == task_map.end());
        task_map.insert({task.get_id(), {std::shared_ptr<Task>(new TaskType(task)), func}});
    }
    auto& get_task_map() {
        return task_map;
    }
    auto get_func(int id) {
        return task_map[id].second;
    }

private:
    std::unordered_map<int, std::pair<std::shared_ptr<Task>, std::function<void(Info)>>> task_map;
};

}  // namespace husky
