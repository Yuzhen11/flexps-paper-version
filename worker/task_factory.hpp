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

    template<typename... Args>
    std::unique_ptr<Task> create_task(Task::Type type, Args&&... args) {
        std::unique_ptr<Task> ptask;
        switch(type) {
            case Task::Type::GenericMLTaskType: {
                ptask.reset(new GenericMLTask(task_id_, std::forward<Args>(args)...));
                break;
            }
            case Task::Type::SingleTaskType: {
                ptask.reset(new SingleTask(task_id_, std::forward<Args>(args)...));
                break;
            }
            case Task::Type::HogwildTaskType: {
                ptask.reset(new HogwildTask(task_id_, std::forward<Args>(args)...));
                break;
            }
            case Task::Type::PSTaskType: {
                ptask.reset(new PSTask(task_id_, std::forward<Args>(args)...));
                break;
            }
            case Task::Type::HuskyTaskType: {
                ptask.reset(new HuskyTask(task_id_, std::forward<Args>(args)...));
                break;
            }
            case Task::Type::BasicTaskType: {
                ptask.reset(new Task(task_id_, std::forward<Args>(args)...));
                break;
            }
            default: {
                throw base::HuskyException("Unknown Task Type!");
            }
        }
        task_id_ += 1;
        return ptask;
    }
    
private:
    int task_id_;
};

}  // namespace husky
