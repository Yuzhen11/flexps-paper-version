#pragma once

#include <memory>
#include <queue>
#include <vector>
#include <sstream>

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
        // Choose a policy to use
        // policy_.reset(new IncreaseBestAmongSomeTriesPolicy());
        policy_.reset(new IncreaseBestPolicy());
        policy_->reset();
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
    
    struct AutoParallelismPolicy {
        int current_worker_per_process;
        int max_worker_per_process;
        int min_worker_per_process;
        std::vector<int> times;
        std::vector<int> worker_per_process_history;
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        int try_iters;
        int current_iters;
        int num_total_iters;
        int sub_epoch;
        bool fixed;  // mark whether the best parallelism degree has found

        virtual ~AutoParallelismPolicy() {}

        /*
         * called for each stage
         */
        void set_num_total_iters(int total_iters) {
            num_total_iters = total_iters;
        }
        /*
         * start timing
         */
        bool start_subepoch() {
            start_time = std::chrono::steady_clock::now();
        }
        /*
         * stop timing for a subepoch
         */
        bool finish_subepoch() {
            auto end_time = std::chrono::steady_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time);
            auto time_int = time.count();
            times.push_back(time_int);
            worker_per_process_history.push_back(current_worker_per_process);
            husky::LOG_I << BLUE("finish subepoch in " + std::to_string(time_int) + " ms");
            sub_epoch += 1;
            current_iters += try_iters;
            return current_iters == num_total_iters;
        }
        /*
         * reset() is called when changing stage 
         */
        void reset() {
            current_iters = 0;
            sub_epoch = 0;
            times.clear();
            worker_per_process_history.clear();
            fixed = false;
            sub_reset();
        }
        /*
         * decide next sub epoch information
         */
        void generate_plan() {
            if (fixed) {
                assert(false);  // Do not suppose to call generate_plan() when the plan is fixed.
            } else {
                sub_generate_plan();
            }
        }

        void print_times_and_history() {
            assert(times.size() == worker_per_process_history.size());
            std::stringstream ss;
            ss << "(worker_per_process, time(ms)) : ";
            for (int i = 0; i < times.size(); ++ i) {
                ss << "(" << std::to_string(worker_per_process_history[i]) 
                    << ", " << std::to_string(times[i]) << "), " ;
            }
            husky::LOG_I << BLUE(ss.str());
        }
        void fix_parallelism() {
            try_iters = num_total_iters - current_iters;
            print_times_and_history();
            fixed = true;
            husky::LOG_I << RED("fix parallelism, worker_per_process: " 
                    << std::to_string(current_worker_per_process));
        }

        // two virtual function
        virtual void sub_reset() = 0;
        virtual void sub_generate_plan() = 0;
    };

    struct IncreaseBestAmongSomeTriesPolicy : public AutoParallelismPolicy {
        virtual void sub_reset() override {
            current_worker_per_process = 0;
            max_worker_per_process = 10;
            min_worker_per_process = 1;
            try_iters = 10;
        }
        virtual void sub_generate_plan() override {
            if (sub_epoch == kNumTries) {
                int min_i = 0;
                int min_time = times[0];
                for (int i = 1; i < times.size(); ++ i) {
                    if (times[i] < min_time) {
                        min_time = times[i];
                        min_i = i;
                    }
                }
                assert(worker_per_process_history.size() > min_i);
                current_worker_per_process = worker_per_process_history[min_i];
                fix_parallelism();
            } else {
                current_worker_per_process += 1;
                if (current_worker_per_process > max_worker_per_process) {
                    current_worker_per_process = max_worker_per_process;
                    husky::LOG_I << RED("Cannot increase current_worker_per_process, the largest is " +
                            std::to_string(current_worker_per_process));
                }
            }
        }
        const int kNumTries = 5;
    };
    struct IncreaseBestPolicy : public AutoParallelismPolicy {
        virtual void sub_reset() override {
            current_worker_per_process = 0;  // TODO(tatiana): better use memory and workload indicator
            max_worker_per_process = 10;
            min_worker_per_process = 1;
            try_iters = 10;
        }
        virtual void sub_generate_plan() override {
            assert(sub_epoch == times.size());
            if (sub_epoch > 1 && times[sub_epoch-1] > times[sub_epoch-2]) {
                current_worker_per_process = worker_per_process_history[sub_epoch-2];
                fix_parallelism();
            } else {
                current_worker_per_process += 1;
                if (current_worker_per_process > max_worker_per_process) {
                    current_worker_per_process = max_worker_per_process;
                    husky::LOG_I << RED("Cannot increase current_worker_per_process, the largest is " +
                            std::to_string(current_worker_per_process));
                }
            }
        }
    };
    std::unique_ptr<AutoParallelismPolicy> policy_;
};

} // namespace husky
