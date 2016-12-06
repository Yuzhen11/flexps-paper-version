#pragma once

#include <cassert>
#include <functional>
#include <unordered_map>
#include <type_traits>
#include "core/task.hpp"
#include "core/info.hpp"

#include "basic.hpp"

namespace husky {

class TaskStore {
public:
    TaskStore() = default;
    /*
     * Add a task into the task_map, the task added should be derived from Task
     */
    template<typename TaskType>
    void add_task(const TaskType& task, const FuncT& func) {
        static_assert(std::is_base_of<Task, TaskType>::value, "TaskType should derived from Task");
        assert(task_map.find(task.get_id()) == task_map.end());
        task_map.insert({task.get_id(), {std::shared_ptr<Task>(new TaskType(task)), func}});
        buffered_tasks.push_back(task.get_id());
    }
    void clear_buffered_tasks() {
        buffered_tasks.clear();
    }
    std::vector<int>& get_buffered_tasks() {
        return buffered_tasks;
    }

    auto& get_task_map() {
        return task_map;
    }
    auto get_func(int id) {
        return task_map[id].second;
    }
    auto get_task(int id) {
        return task_map[id].first;
    }

private:
    std::unordered_map<int, std::pair<std::shared_ptr<Task>, FuncT>> task_map;
    std::vector<int> buffered_tasks;
};

}  // namespace husky
