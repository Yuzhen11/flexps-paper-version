#pragma once

#include "core/task.hpp"

namespace husky {

/*
 * TaskFactory is to create tasks for users
 *
 * It's a singleton
 */
class TaskFactory {
   public:
    static TaskFactory& Get() {
        static TaskFactory task_factory;
        return task_factory;
    }

    template <typename TaskT, typename... Args>
    TaskT CreateTask(Args&&... args) {
        TaskT task(task_id_, std::forward<Args>(args)...);
        task_id_ += 1;
        return task;
    }

   private:
    int task_id_;
};

}  // namespace husky
