#pragma once

#include <ctime>
#include <cstdlib>
#include <unordered_map>
#include <queue>
#include "core/task.hpp"
#include "core/instance.hpp"

namespace husky {
class TaskScheduler {
public:
    TaskScheduler(WorkerInfo& worker_info_)
        : worker_info(worker_info_) {
    }
    virtual void init_tasks(const std::vector<std::shared_ptr<Task>>&) = 0;
    virtual void finish_local_instance(int instance_id, int proc_id) = 0;
    virtual std::vector<std::shared_ptr<Instance>> extract_instances() = 0;
    virtual bool is_finished() = 0;

    virtual ~TaskScheduler() {};
protected:
    WorkerInfo& worker_info;
};

class SequentialTaskScheduler : public TaskScheduler {
public:
    SequentialTaskScheduler(WorkerInfo& worker_info_)
        : TaskScheduler(worker_info_) {
    }

    virtual ~SequentialTaskScheduler() override {}

    virtual void init_tasks(const std::vector<std::shared_ptr<Task>>& tasks) override {
        for (auto& task : tasks) {
            tasks_queue.push(task);
        }
    }
    virtual void finish_local_instance(int instance_id, int proc_id) override {
        tracker.erase(proc_id);  // Mark a process to finished
        if (tracker.size() == 0) {  // If all the processes are done, the instance is done
            auto& task = tasks_queue.front();
            task->inc_epoch();  // Trying to work on next epoch
            if (task->get_current_epoch() == task->get_total_epoch()) {  // If all the epochs are done, then task is done
                tasks_queue.pop();
            }
        }
    }
    virtual std::vector<std::shared_ptr<Instance>> extract_instances() override {
        // Assign no instance if 1. task_queue is empty or 2. current instance is still running 
        if (tasks_queue.empty() || !tracker.empty())   
            return {};
        auto& task = tasks_queue.front();
        auto instance = task_to_instance(*task);
        init_tracker(instance);
        return {instance};
    }
    virtual bool is_finished() override {
        return tasks_queue.empty();
    }

protected:
    std::queue<std::shared_ptr<Task>> tasks_queue;
    std::unordered_set<int> tracker;
private:
    void init_tracker(const std::shared_ptr<Instance>& instance) {
        auto& cluster = instance->get_cluster();
        for (auto kv : cluster) {
            tracker.insert(kv.first);
        }
    }
    std::shared_ptr<Instance> task_to_instance(Task& task) {
        auto num_workers = worker_info.get_num_workers();
        // TODO: For debug and testing only. Master needs to design workers number for GenericMLTaskType
        if (task.get_type() == Task::Type::GenericMLTaskType)
            task.set_num_workers(1);
        assert(num_workers >= task.get_num_workers());
        // randomly select threads 
        std::vector<int> selected_workers;
        while (selected_workers.size() < task.get_num_workers()) {
            int tid = rand()%num_workers;
            while (std::find(selected_workers.begin(), selected_workers.end(), tid) != selected_workers.end()) {
                tid = rand()%num_workers;
            }
            selected_workers.push_back(tid);
        }
        // create the instance
        std::shared_ptr<Instance> instance(new Instance);
        // TODO If the task type is GenericMLTaskType, need to decide it's real running type now
        if (task.get_type() == Task::Type::GenericMLTaskType) {
            // TODO now set to SingleTaskType for testing...
            instance->set_task(task, Task::Type::SingleTaskType);
            // instance->set_task(task, Task::Type::HogwildTaskType);
        } else {
            instance->set_task(task);
        }
        for (int i = 0; i < selected_workers.size(); ++i) {
            int proc_id = worker_info.get_proc_id(selected_workers[i]);
            instance->add_thread(proc_id, selected_workers[i], i);
        }
        return instance;
    }
};
}  // namespace husky
