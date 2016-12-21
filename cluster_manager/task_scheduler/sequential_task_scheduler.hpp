#pragma once

#include "task_scheduler.hpp"
#include "task_scheduler_utils.hpp"

namespace husky {

/*
 * SequentialTaskScheduler implements TaskScheduler
 *
 * It basically extracts and executes tasks from queue one by one
 */
class SequentialTaskScheduler : public TaskScheduler {
   public:
    SequentialTaskScheduler(WorkerInfo& worker_info_) : TaskScheduler(worker_info_) {}

    virtual ~SequentialTaskScheduler() override {}

    virtual void init_tasks(const std::vector<std::shared_ptr<Task>>& tasks) override {
        for (auto& task : tasks) {
            tasks_queue.push(task);
        }
    }
    virtual void finish_local_instance(int instance_id, int proc_id) override {
        tracker.erase(proc_id);     // Mark a process to finished
        if (tracker.size() == 0) {  // If all the processes are done, the instance is done
            auto& task = tasks_queue.front();
            task->inc_epoch();  // Trying to work on next epoch
            if (task->get_current_epoch() ==
                task->get_total_epoch()) {  // If all the epochs are done, then task is done
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
    virtual bool is_finished() override { return tasks_queue.empty(); }

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

    /*
     * Get workers globally
     *
     * Randomly select workers globally
     */
    std::vector<int> get_workers(int required_num_workers) {
        std::vector<int> selected_workers;
        int num_workers = worker_info.get_num_workers();
        assert(num_workers >= required_num_workers);
        while (selected_workers.size() < required_num_workers) {
            int tid = rand() % num_workers;
            while (std::find(selected_workers.begin(), selected_workers.end(), tid) != selected_workers.end()) {
                tid = rand() % num_workers;
            }
            selected_workers.push_back(tid);
        }
        return selected_workers;
    }

    /*
     * Get workers locally, only used for Task::Type::HogwildTaskType
     *
     * Randomly select workers locally within one process
     */
    std::vector<int> get_local_workers(int required_num_workers) {
        std::vector<int> selected_workers;
        // To be simple, just randomly select a process
        std::vector<int> pids = worker_info.get_pids();
        int pid = rand() % pids.size();
        int num_local_workers = worker_info.get_num_local_workers(pid);
        assert(num_local_workers >= required_num_workers);
        while (selected_workers.size() < required_num_workers) {
            int tid = worker_info.local_to_global_id(pid, rand() % num_local_workers);
            while (std::find(selected_workers.begin(), selected_workers.end(), tid) != selected_workers.end()) {
                tid = worker_info.local_to_global_id(pid, rand() % num_local_workers);
            }
            selected_workers.push_back(tid);
        }
        return selected_workers;
    }

    /*
     * Generate real running instance from task
     *
     * Be careful that instance may not be complete within the process:
     * Empty instance -> set_task() -> set_num_workers() -> add_thread() -> complete instance
     */
    std::shared_ptr<Instance> task_to_instance(const Task& task) {
        // create the instance
        std::shared_ptr<Instance> instance(new Instance);
        instance_basic_setup(instance, task);

        // randomly select threads
        std::vector<int> selected_workers;
        if (instance->get_type() == Task::Type::HogwildTaskType)
            selected_workers = get_local_workers(instance->get_num_workers());
        else
            selected_workers = get_workers(instance->get_num_workers());

        // add threads to instance
        for (int i = 0; i < selected_workers.size(); ++i) {
            int proc_id = worker_info.get_process_id(selected_workers[i]);
            instance->add_thread(proc_id, selected_workers[i], i);
        }
        return instance;
    }
};

}  // namespace husky
