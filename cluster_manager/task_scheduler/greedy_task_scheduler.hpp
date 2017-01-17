#pragma once

#include <iostream>
#include <unordered_set>

#include "task_scheduler.hpp"
#include "task_scheduler_utils.hpp"

#include "core/color.hpp"

namespace husky {

struct PairHash {
    template<typename T, typename U>
    std::size_t operator()(const std::pair<T, U>& x) const {
        return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
    }
};

/*
 * AvailableWorkers keep track of the available workers
 */
class AvailableWorkers {
   public:
    void add_worker(int pid, int tid) {
        workers_.insert({tid, pid});
        pid_tids_[pid].insert(tid);
    }

    void remove_worker(int pid, int tid) {
        workers_.erase({tid, pid});
        pid_tids_[pid].erase(tid);
    }

    // return all available workers in the cluster
    std::vector<std::pair<int, int>> get_thread_per_worker(int num_thread_per_worker, int num_processes) {
        std::map<int, std::vector<int>> pid_tids_map; 
        std::vector<int> tids;
        // init map
        for (int i = 0; i < num_processes; i++) {
            pid_tids_map.emplace(i, tids);
        }

        for (auto tid_pid : workers_) {
            pid_tids_map[tid_pid.second].push_back(tid_pid.first);
        }

        // guarantee at least thread_per_worker thread in each worker
        int guarantee = 1;
        for(auto pid_tid_map : pid_tids_map) {
            if (pid_tid_map.second.size() < num_thread_per_worker) {
                guarantee = 0;
                break;
            }
        }
        // current availworkers can guarantee, get these threads
        if (guarantee == 1) {
            std::vector<std::pair<int,int>> selected_workers;
            for (auto pid_tid_map : pid_tids_map) {
                for (int j = 0; j < num_thread_per_worker; j++) {
                    selected_workers.push_back({pid_tid_map.first, pid_tid_map.second[j]});  // <pid, tid>
                }
            }

            // erase from workers
            for (auto pid_tid : selected_workers) {
                workers_.erase({pid_tid.second, pid_tid.first});
            }
            //erase from pid_tids
            for (auto pid_tid : selected_workers)
                pid_tids_[pid_tid.first].erase(pid_tid.second);

            return selected_workers;
        }

        return {};
    }
    /*
     * @return { <pid, tid> }
     */
    std::vector<std::pair<int,int>> get_workers(int required_num_workers) {
        assert(workers_.size() >= required_num_workers);
        // Since all the workers are stored in unordered_set, 
        // I just get the first *required_num_workers* threads
        std::vector<std::pair<int,int>> selected_workers;
        auto it = workers_.begin();
        while (selected_workers.size() < required_num_workers) {
            selected_workers.push_back({it->second, it->first});  // <pid, tid>
            ++it;
        }
        // erase from workers_
        workers_.erase(workers_.begin(), it);
        // erase from pid_tids_
        for (auto pid_tid : selected_workers)
            pid_tids_[pid_tid.first].erase(pid_tid.second);
        return selected_workers;
    }

    std::vector<std::pair<int,int>> get_local_workers(int required_num_workers) {
        // find the first process that contains no less than *required_num_workers* 's threads
        for (auto& kv : pid_tids_) {
            if (kv.second.size() >= required_num_workers) {
                std::vector<std::pair<int,int>> selected_workers;
                auto it = kv.second.begin();
                while (selected_workers.size() < required_num_workers) {
                    selected_workers.push_back({kv.first, *it});
                    ++it;
                }
                // erase from workers
                for (auto pid_tid : selected_workers) {
                    workers_.erase({pid_tid.second, pid_tid.first});
                }
                // erase from pid_tids_
                kv.second.erase(kv.second.begin(), it);
                return selected_workers;
            }
        }
        return {};
    }

    int get_num_available_workers() {
        return workers_.size();
    }

    int get_max_local_workers() {
        int max_num = 0;
        for (auto& kv : pid_tids_)
            max_num = std::max(max_num, static_cast<int>(kv.second.size()));
        return max_num;
    }
   private:
    std::unordered_set<std::pair<int,int>, PairHash> workers_;  // <tid, pid>
    std::unordered_map<int, std::unordered_set<int>> pid_tids_;
};

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
        for (int i = 0; i < tasks_.size(); ++ i) {
            if (task_status_.at(i) == 0) {  // ready to run
                // create the instance
                std::shared_ptr<Instance> instance(new Instance);
                instance_basic_setup(instance, *tasks_[i]);

                if (instance->get_type() == Task::Type::TwoPhasesTaskType) {
                    if (instance->get_epoch() % 2 == 1) {
                        //  make sure per worker has n threads available
                        int thread_per_worker = instance->get_num_workers();
                        if (available_workers_.get_num_available_workers() >= thread_per_worker * num_processes_) {
                            auto pid_tids = available_workers_.get_thread_per_worker(thread_per_worker, num_processes_);
                            if (pid_tids.size() > 0) {
                                int j = 0;
                                for (auto pid_tid : pid_tids) {
                                    instance->add_thread(pid_tid.first, pid_tid.second, j++);
                                    tracker_[instance->get_id()].insert(pid_tid.first);
                                    task_id_pid_tids_[{instance->get_id(), pid_tid.first}].push_back(pid_tid.second);
                                }
                                instances.push_back(std::move(instance));
                                task_status_.at(i) = 1;
                            }
                            else {
                                continue;
                            }
                        }
                        else {
                            continue;
                        }
                    }
                    else {
                        // run even epoch with 1 thread default
                        auto pid_tids = available_workers_.get_workers(1);
                        if (pid_tids.size() > 0) {
                            int j = 0;
                            for (auto pid_tid : pid_tids) {
                                instance->add_thread(pid_tid.first, pid_tid.second, j++);
                                tracker_[instance->get_id()].insert(pid_tid.first);
                                task_id_pid_tids_[{instance->get_id(), pid_tid.first}].push_back(pid_tid.second);
                            }
                            instances.push_back(std::move(instance));
                            task_status_.at(i) = 1;
                        }
                        else {
                            continue;
                        }
                    }
                }
                else if (instance->get_type() == Task::Type::HogwildTaskType) {
                    if (available_workers_.get_max_local_workers() >= instance->get_num_workers()) {
                        // extract from local_workers
                        auto pid_tids = available_workers_.get_local_workers(instance->get_num_workers());
                        int j = 0;
                        for (auto pid_tid : pid_tids) {
                            instance->add_thread(pid_tid.first, pid_tid.second, j++);
                            tracker_[instance->get_id()].insert(pid_tid.first);
                            task_id_pid_tids_[{instance->get_id(), pid_tid.first}].push_back(pid_tid.second);
                        }
                        instances.push_back(std::move(instance));
                        task_status_.at(i) = 1;
                    } else {
                        continue;
                    }
                }
                else {
                    if (available_workers_.get_num_available_workers() >= instance->get_num_workers()) {
                        // extract from global workers
                        auto pid_tids = available_workers_.get_workers(instance->get_num_workers());
                        int j = 0;
                        for (auto pid_tid : pid_tids) {
                            instance->add_thread(pid_tid.first, pid_tid.second, j++);
                            tracker_[instance->get_id()].insert(pid_tid.first);
                            task_id_pid_tids_[{instance->get_id(), pid_tid.first}].push_back(pid_tid.second);
                        }
                        instances.push_back(std::move(instance));
                        task_status_.at(i) = 1;
                    } else {
                        continue;
                    }
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
    int num_processes_;     // num workers(machines) in cluster
    std::vector<std::shared_ptr<Task>> tasks_;
    std::vector<int> task_status_;  // 0: ready to run, 1: running, 2: done
    AvailableWorkers available_workers_;
    std::unordered_map<int, std::unordered_set<int>> tracker_;   // { task_id1:{pid1...}, { task_id2:{..}}, ...}
    std::unordered_map<std::pair<int,int>, std::vector<int>, PairHash> task_id_pid_tids_;   // <task_id, pid> : {tid1, tid2...}

};


}  // namespace husky
