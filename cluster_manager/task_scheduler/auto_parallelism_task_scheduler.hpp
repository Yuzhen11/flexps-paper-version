#pragma once

#include <memory>
#include <queue>
#include <vector>

#include "cluster_manager/task_scheduler/available_workers.hpp"
#include "cluster_manager/task_scheduler/task_scheduler.hpp"
#include "core/instance.hpp"
#include "core/task.hpp"

#include <chrono>

namespace husky {

/*
 * Sequential and auto parallelism
 */
class AutoParallelismTaskScheduler : public TaskScheduler {
   public:
    AutoParallelismTaskScheduler(WorkerInfo& worker_info_) : TaskScheduler(worker_info_) {
        num_processes_ = worker_info_.get_num_processes();
        // initialize the available_workers_
        auto tids = worker_info_.get_global_tids();
        for (auto tid : tids) {
            available_workers_.add_worker(worker_info_.get_process_id(tid), tid);
        }
    }

    virtual ~AutoParallelismTaskScheduler() override {}

    virtual void init_tasks(const std::vector<std::shared_ptr<Task>>& tasks) override; 

    virtual void finish_thread(int instance_id, int global_thread_id) override;

    virtual std::vector<std::shared_ptr<Instance>> extract_instances() override;

    virtual bool is_finished() override;

   private:

    /*
     * Generate real running instance from task
     *
     * Be careful that instance may not be complete within the process:
     * Empty instance -> set_task() -> set_num_workers() -> add_thread() -> complete instance
     */
    std::shared_ptr<Instance> task_to_instance(const Task& task);

    std::shared_ptr<Instance> task_to_instance_auto_parallelism(const Task& task);

   private:
    std::queue<std::shared_ptr<Task>> tasks_queue_;
    AvailableWorkers available_workers_;
    int num_processes_;  // num of machines in cluster
    std::unordered_map<int, std::unordered_map<int, std::unordered_set<int>>> 
        task_id_pid_tids_;  // task_id : pid : {tid1, tid2...}
    
    struct SearchWorker {
        int current_worker_per_process = 0;
        int max_worker_per_process = 10;
        int min_worker_per_process = 1;
        std::vector<std::chrono::microseconds> times;
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        int try_iters = 10;
        int current_iters = 0;
        int num_total_iters;
        int sub_epoch = 0;

        void set_num_total_iters(int total_iters) {
            num_total_iters = total_iters;
        }
        void reset() {
            current_worker_per_process = 0;
            max_worker_per_process = 10;
            min_worker_per_process = 1;
            try_iters = 10;
            current_iters = 0;
            sub_epoch = 0;
        }
        bool start_subepoch() {
            start_time = std::chrono::steady_clock::now();
        }
        bool finish_subepoch() {
            auto end_time = std::chrono::steady_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time);
            times.push_back(time);
            husky::LOG_I << BLUE("finish subepoch in " + std::to_string(time.count()) + " ms");
            sub_epoch += 1;
            current_iters += try_iters;
            return current_iters == num_total_iters;
        }
        void generate_plan() {
            current_worker_per_process += 1;
            assert(current_worker_per_process <= max_worker_per_process);
            // TODO some strategies?
        }
    };
    SearchWorker search_worker_;
};

} // namespace husky
