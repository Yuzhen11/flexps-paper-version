#pragma once

#include <cassert>
#include <functional>
#include <type_traits>
#include <unordered_map>
#include "core/info.hpp"
#include "core/task.hpp"

#include "basic.hpp"

namespace husky {

class TaskStore {
   public:
    TaskStore() = default;
    /*
     * Add a task into the task_map, the task added should be derived from Task
     */
    template<typename TaskT>
    void add_task(const TaskT& task, const FuncT& func) {
        std::unique_ptr<Task> ptask(new TaskT(task));
        int tid = ptask->get_id();
        assert(task_map.find(tid) == task_map.end());
        task_map.insert(std::make_pair(tid, std::make_pair(std::move(ptask), func)));
        buffered_tasks.push_back(tid);
    }
    void clear_buffered_tasks() { buffered_tasks.clear(); }
    std::vector<int>& get_buffered_tasks() { return buffered_tasks; }

    auto& get_task_map() { return task_map; }
    auto get_func(int id) { return task_map[id].second; }
    std::unique_ptr<Task>& get_task(int id) { return task_map[id].first; }

   private:
    std::unordered_map<int, std::pair<std::unique_ptr<Task>, FuncT>> task_map;
    std::vector<int> buffered_tasks;
};

}  // namespace husky
