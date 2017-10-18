#include "cluster_manager/task_scheduler/auto_parallelism_task_scheduler.hpp"

#include <memory>
#include <queue>
#include <vector>

#include "cluster_manager/task_scheduler/available_workers.hpp"
#include "cluster_manager/task_scheduler/task_scheduler.hpp"
#include "cluster_manager/task_scheduler/task_scheduler_utils.hpp"
#include "cluster_manager/task_scheduler/history_manager.hpp"

namespace husky {

void AutoParallelismTaskScheduler::init_tasks(const std::vector<std::shared_ptr<Task>>& tasks) {
    bool first = true;
    for (auto& task : tasks) {
        tasks_queue_.push(task);
        if (task->get_type() == Task::Type::AutoParallelismTaskType) {
            first = false;
            auto& epoch_iters = static_cast<AutoParallelismTask*>(task.get())->get_epoch_iters();
            assert(epoch_iters.size() > 0);
            search_worker_.set_num_total_iters(epoch_iters[0]);  // initialize search_worker_
        }
    }
    assert(!first);
}

void AutoParallelismTaskScheduler::finish_thread(int instance_id, int global_thread_id) {
    int proc_id = worker_info.get_process_id(global_thread_id);
    auto& threads = task_id_pid_tids_[instance_id][proc_id];
    // remove from task_id_pid_tids_
    threads.erase(global_thread_id);
    available_workers_.add_worker(proc_id, global_thread_id);
    if (threads.empty()) {
        task_id_pid_tids_[instance_id].erase(proc_id);
        if (task_id_pid_tids_[instance_id].size() == 0) {  // If all the processes are done, the instance is done
            task_id_pid_tids_.erase(instance_id);

            auto& task = tasks_queue_.front();
            if (task->get_type() == Task::Type::AutoParallelismTaskType) {
                husky::LOG_I << CLAY("Task " + std::to_string(task->get_id()) + 
                        " epoch " + std::to_string(task->get_current_epoch()) + 
                        " subepoch " + std::to_string(search_worker_.sub_epoch) + 
                        " finished ");
                bool finish = search_worker_.finish_subepoch();
                if (finish) {  // finish stage
                    search_worker_.reset();
                    husky::LOG_I << CLAY("Task " + std::to_string(task->get_id()) + " epoch " + std::to_string(task->get_current_epoch()) + " finished ");
                    task->inc_epoch();  // Trying to work on next epoch
                    if (task->get_current_epoch() ==
                        task->get_total_epoch()) {  // If all the epochs are done, then task is done
                        tasks_queue_.pop();
                        // reset num total iters for next task
                        if (!tasks_queue_.empty() && tasks_queue_.front()->get_type() == Task::Type::AutoParallelismTaskType) {
                            auto& epoch_iters = static_cast<AutoParallelismTask*>(tasks_queue_.front().get())->get_epoch_iters();
                            assert(epoch_iters.size() > 0);
                            search_worker_.set_num_total_iters(epoch_iters[0]);  // initialize search_worker_
                        }
                    } else {
                        // reset num total iters for changing stage
                        auto& epoch_iters = static_cast<AutoParallelismTask*>(task.get())->get_epoch_iters();
                        assert(epoch_iters.size() > task->get_current_epoch());
                        search_worker_.set_num_total_iters(epoch_iters[task->get_current_epoch()]);  // initialize search_worker_
                    }
                }
            } else {
                husky::LOG_I << CLAY("Task " + std::to_string(task->get_id()) + " epoch " + std::to_string(task->get_current_epoch()) + " finished ");
                task->inc_epoch();  // Trying to work on next epoch
                if (task->get_current_epoch() ==
                    task->get_total_epoch()) {  // If all the epochs are done, then task is done
                    tasks_queue_.pop();
                    // reset num total iters
                    if (!tasks_queue_.empty() && tasks_queue_.front()->get_type() == Task::Type::AutoParallelismTaskType) {
                        auto& epoch_iters = static_cast<AutoParallelismTask*>(tasks_queue_.front().get())->get_epoch_iters();
                        assert(epoch_iters.size() > 0);
                        search_worker_.set_num_total_iters(epoch_iters[0]);  // initialize search_worker_
                    }
                }
            }
        }
    }
}

std::vector<std::shared_ptr<Instance>> AutoParallelismTaskScheduler::extract_instances() {
    // Assign no instance if 1. task_queue is empty or 2. current instance is still running
    if (tasks_queue_.empty() || !task_id_pid_tids_.empty())
        return {};
    auto& task = tasks_queue_.front();
    if (task->get_type() == Task::Type::AutoParallelismTaskType) {
        auto instance = task_to_instance_auto_parallelism(*task);
        return {instance};
    } else {
        auto instance = task_to_instance(*task);
        return {instance};
    }
}

bool AutoParallelismTaskScheduler::is_finished() { return tasks_queue_.empty(); }

std::shared_ptr<Instance> AutoParallelismTaskScheduler::task_to_instance_auto_parallelism(const Task& task) {
    search_worker_.generate_plan();
    std::shared_ptr<Instance> instance(new Instance);
    instance->set_task(task);
    instance->set_num_workers(search_worker_.current_worker_per_process * num_processes_);

    std::vector<std::pair<int, int>> pid_tids = available_workers_.get_workers_per_process(search_worker_.current_worker_per_process, num_processes_);
    assert(!pid_tids.empty());
    int j = 0;
    for (auto pid_tid : pid_tids) {
        instance->add_thread(pid_tid.first, pid_tid.second, j++);
        task_id_pid_tids_[instance->get_id()][pid_tid.first].insert(pid_tid.second);
    }
    // update history
    HistoryManager::get().update_history(instance->get_id(), pid_tids);
    husky::LOG_I << YELLOW("Task: "+std::to_string(instance->get_id())+" added");
    husky::LOG_I << YELLOW("num_workers: " + std::to_string(instance->get_num_workers())
            + "\ncurrent_iters: " + std::to_string(search_worker_.current_iters)
            + "\ntotal_iters: " << std::to_string(search_worker_.num_total_iters)
            + "\ntry_iters: " << std::to_string(search_worker_.try_iters));

    auto* p_task = instance->get_task();
    static_cast<AutoParallelismTask*>(p_task)->set_current_stage_iters(search_worker_.try_iters);
    search_worker_.start_subepoch();
    return instance;
}

std::shared_ptr<Instance> AutoParallelismTaskScheduler::task_to_instance(const Task& task) {
    // create the instance
    std::shared_ptr<Instance> instance(new Instance);
    instance_basic_setup(instance, task);

    int required_num_threads = instance->get_num_workers();
    std::vector<int> candidate_proc = get_preferred_proc(instance->get_id());
    // select threads according to the instance
    
    std::vector<std::pair<int,int>> pid_tids = select_threads_from_subset(instance, available_workers_, num_processes_, 
            required_num_threads, candidate_proc);

    // If requirement is satisfied, set the instance
    if (!pid_tids.empty()) {
        int j = 0;
        for (auto pid_tid : pid_tids) {
            instance->add_thread(pid_tid.first, pid_tid.second, j++);
            task_id_pid_tids_[instance->get_id()][pid_tid.first].insert(pid_tid.second);
        }
        // update history
        HistoryManager::get().update_history(instance->get_id(), pid_tids);
        husky::LOG_I << YELLOW("Task: "+std::to_string(instance->get_id())+" added");
    } else {
        throw base::HuskyException("[AutoParallelism Task Scheduler] Cannot assign next instance");
    }
    return instance;
}

} // namespace husky
