#pragma once

#include "core/instance.hpp"
#include "core/task.hpp"

namespace husky {
namespace {

void instance_basic_setup(std::shared_ptr<Instance>& instance, const Task& task) {
    // TODO If the task type is GenericMLTaskType and the running type is unset,
    // need to decide it's real running type now
    if (task.get_type() == Task::Type::GenericMLTaskType &&
        static_cast<const GenericMLTask&>(task).get_running_type() != Task::Type::DummyType) {
        // TODO now set to SingleTaskType for testing...
        instance->set_task(task, Task::Type::SingleTaskType);
        // instance->set_task(task, Task::Type::HogwildTaskType);
    } else {
        instance->set_task(task);
    }

    // TODO: ClusterManager needs to design workers number for GenericMLTaskType if user didn't set it
    if (task.get_type() == Task::Type::GenericMLTaskType && task.get_num_workers() == 0)
        instance->set_num_workers(1);
}

}  // namespace anonymous
}  // namespace husky
