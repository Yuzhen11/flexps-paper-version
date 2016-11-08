#pragma once

#include <ctime>
#include <cstdlib>
#include <unordered_map>
#include <queue>
#include "core/common/task.hpp"
#include "core/common/instance.hpp"

namespace husky {
class TaskScheduler {
public:
    TaskScheduler(WorkerInfo& worker_info_)
        : worker_info(worker_info_) {
    }
    virtual void init_tasks(const std::vector<Task>&) = 0;
    virtual void finish_local_instance(int instance_id, int proc_id) = 0;
    virtual std::vector<Instance> extract_instances() = 0;
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

    virtual void init_tasks(const std::vector<Task>& tasks) override {
        for (auto& task : tasks) {
            tasks_queue.push(task);
        }
    }
    virtual void finish_local_instance(int instance_id, int proc_id) override {
        tracker.erase(proc_id);
        if (tracker.size() == 0) {
            // instance done
            auto& task = tasks_queue.front();
            task.current_epoch += 1;
            if (task.current_epoch == task.total_epoch) {
                tasks_queue.pop();
            }
        }
    }
    virtual std::vector<Instance> extract_instances() override {
        if (tasks_queue.empty())
            return {};
        auto& task = tasks_queue.front();
        auto instance = task_to_instance(task);
        init_tracker(instance);
        return {instance};
    }
    virtual bool is_finished() override {
        return tasks_queue.empty();
    }

protected:
    std::queue<Task> tasks_queue;
    std::unordered_set<int> tracker;
private:
    void init_tracker(const Instance& instance) {
        auto& cluster = instance.get_cluster();
        for (auto kv : cluster) {
            tracker.insert(kv.first);
        }
    }
    Instance task_to_instance(const Task& task) {
        auto num_workers = worker_info.get_num_workers();
        assert(num_workers >= task.num_workers);
        // randomly select threads 
        std::vector<int> selected_workers;
        while (selected_workers.size() < task.num_workers) {
            int tid = rand()%num_workers;
            while (std::find(selected_workers.begin(), selected_workers.end(), tid) != selected_workers.end()) {
                tid = rand()%num_workers;
            }
            selected_workers.push_back(tid);
        }
        // create the instance
        Instance instance(task.id, task.current_epoch);
        for (auto tid : selected_workers) {
            int proc_id = worker_info.get_proc_id(tid);
            instance.add_thread(proc_id, tid);
        }
        return instance;
    }
};
}  // namespace husky
