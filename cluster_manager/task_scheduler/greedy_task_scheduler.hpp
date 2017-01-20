#pragma once

#include <iostream>
#include <unordered_set>

#include "available_workers.hpp"
#include "task_scheduler.hpp"
#include "task_scheduler_utils.hpp"

#include "core/color.hpp"


namespace husky {

/*
 * GreedyTaskScheduler implements TaskScheduler
 *
 * It can disitribute more than one task at each time
 */
class GreedyTaskScheduler : public TaskScheduler {
   public:
    GreedyTaskScheduler(WorkerInfo& worker_info_) : TaskScheduler(worker_info_) {
        num_processes_ = worker_info_.get_num_processes();
        // initialize the available_workers_
        auto tids = worker_info_.get_global_tids();
        for (auto tid : tids) {
            available_workers_.add_worker(worker_info_.get_process_id(tid), tid);
        }
    }

    virtual ~GreedyTaskScheduler() override {}

    virtual void init_tasks(const std::vector<std::shared_ptr<Task>>& tasks) override {
        tasks_ = tasks;
        task_status_.clear();
        task_status_.resize(tasks_.size(), 0);
    }

    virtual void finish_local_instance(int instance_id, int proc_id) override {
        // remove from tracker_
        tracker_[instance_id].erase(proc_id);
        if (tracker_[instance_id].empty()) {  // all the proc_id are done
            tracker_.erase(instance_id);
            // linear search to find the task_id
            auto p = std::find_if(tasks_.begin(), tasks_.end(), [instance_id](const std::shared_ptr<Task>& task){
                return task->get_id() == instance_id;
            });
            auto& task = *p;
            task->inc_epoch();
            int idx = distance(tasks_.begin(), p);
            if (task->get_current_epoch() == task->get_total_epoch()) {
                task_status_.at(idx) = 2;  // mark it as finished
            } else {
                task_status_.at(idx) = 0;  // mark it as ready
            }
        }
        auto& tids = task_id_pid_tids_[{instance_id, proc_id}];
        for (auto tid : tids) {
            // add to available_workers_
            available_workers_.add_worker(proc_id, tid);
        }
        // remove from task_id_pid_tids_
        task_id_pid_tids_.erase({instance_id, proc_id});
    }

    virtual std::vector<std::shared_ptr<Instance>> extract_instances() override {
        std::vector<std::shared_ptr<Instance>> instances;
        // Go through the tasks list once and assign as much as possible
        for (int i = 0; i < tasks_.size(); ++ i) {
            if (task_status_.at(i) == 0) {  // ready to run
                // create the instance
                std::shared_ptr<Instance> instance(new Instance);
                instance_basic_setup(instance, *tasks_[i]);

                std::vector<std::pair<int,int>> pid_tids;
                if ((instance->get_type() == Task::Type::TwoPhasesTaskType && instance->get_epoch() % 2 == 0)
                    || instance->get_type() == Task::Type::FixedWorkersTaskType) {
                    int thread_per_worker = instance->get_num_workers();
                    pid_tids = available_workers_.get_workers_per_process(thread_per_worker, num_processes_);
                }
                else if (instance->get_type() == Task::Type::TwoPhasesTaskType && instance->get_epoch() % 2 == 1) {
                    // run even epoch with 1 thread default
                    pid_tids = available_workers_.get_workers(1);
                }
                else if (instance->get_type() == Task::Type::HogwildTaskType) {
                    // extract from local_workers
                    pid_tids = available_workers_.get_local_workers(instance->get_num_workers());
                }
                else {
                    // extract from global workers
                    pid_tids = available_workers_.get_workers(instance->get_num_workers());
                }
                // If requirement is satisfied, add the instance
                if (!pid_tids.empty()) {
                    int j = 0;
                    for (auto pid_tid : pid_tids) {
                        instance->add_thread(pid_tid.first, pid_tid.second, j++);
                        tracker_[instance->get_id()].insert(pid_tid.first);
                        task_id_pid_tids_[{instance->get_id(), pid_tid.first}].push_back(pid_tid.second);
                    }
                    instances.push_back(std::move(instance));
                    task_status_.at(i) = 1;
                }
            }
        }
        return instances;
    }

    virtual bool is_finished() override {
        for (auto status : task_status_) {
            if (status != 2)
                return false;
        }
        return true;
    }
   private:
    // for debug
    void print_available_workers() {
        std::cout << "available_workers_: " << available_workers_.get_num_available_workers() << std::endl;
    }
    void print_task_status() {
        std::cout << "task_scheduler: " << std::endl;
        for (auto status : task_status_) {
            std::cout << status << " ";
        }
        std::cout << std::endl;
    }
   private:
    int num_processes_;     // num of machines in cluster
    std::vector<std::shared_ptr<Task>> tasks_;
    std::vector<int> task_status_;  // 0: ready to run, 1: running, 2: done
    AvailableWorkers available_workers_;
    std::unordered_map<int, std::unordered_set<int>> tracker_;   // { task_id1:{pid1...}, { task_id2:{..}}, ...}
    std::unordered_map<std::pair<int,int>, std::vector<int>, PairHash> task_id_pid_tids_;   // <task_id, pid> : {tid1, tid2...}

};


}  // namespace husky
