#include "cluster_manager/task_scheduler/priority_task_scheduler.hpp"

#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "cluster_manager/task_scheduler/available_workers.hpp"
#include "cluster_manager/task_scheduler/task_manager.hpp"
#include "cluster_manager/task_scheduler/task_scheduler.hpp"
#include "cluster_manager/task_scheduler/task_scheduler_utils.hpp"
#include "cluster_manager/task_scheduler/history_manager.hpp"

namespace husky {

void PriorityTaskScheduler::init_tasks(const std::vector<std::shared_ptr<Task>>& tasks) {
    task_manager_.add_tasks(tasks);
}

void PriorityTaskScheduler::finish_thread(int instance_id, int global_thread_id) {
    int proc_id = worker_info.get_process_id(global_thread_id);
    available_workers_.add_worker(proc_id, global_thread_id);
    task_manager_.finish_thread(instance_id, proc_id, global_thread_id);

    // available_workers_.print_available_workers();
}

std::vector<std::shared_ptr<Instance>> PriorityTaskScheduler::extract_instances() {
    std::vector<std::shared_ptr<Instance>> instances;
    std::unordered_set<int>  process_lock;
    // 1. Handle angry tasks first, lock preferred processes if the task is not scheduled
    if (task_manager_.exist_angry_tasks()) {
        husky::DLOG_I << "angry";
        auto begin = task_manager_.angry_list_begin();
        auto end = task_manager_.angry_list_end();
        std::vector<int> success_tasks;
        std::vector<int> fail_tasks;
        // std::cout << "extract instances invoked" << std::endl;
        while (begin != end) {
            int id = *begin;
            // std::cout << "processing id in angry list: "  << id << std::endl;
            // available_workers_.print_available_workers();

            std::shared_ptr<Instance> instance(new Instance);
            instance_basic_setup(instance, *(task_manager_.get_task_by_id(id)));


            std::vector<int> proc_ids;

            auto task = task_manager_.get_task_by_id(id);
            auto& hint = task->get_hint();
            if (task->get_type() == Task::Type::MLTaskType 
                    && hint.at(husky::constants::kType) == husky::constants::kPS) {
                // don't consider history when running pstask
                for (int i = 0; i < num_processes_; ++ i) {
                    proc_ids.push_back(i);
                }
            } else {
                proc_ids = task_manager_.get_preferred_proc(id);
            }

            std::vector<int> candidate_pids;
            for (auto& pid : proc_ids) {
                if (process_lock.find(pid) == process_lock.end()) {
                    candidate_pids.push_back(pid);
                }
            }
            /*
            std::cout << "prefered pids: ";
            for (auto pid: proc_ids)
                std::cout << pid;
            std::cout << std::endl;

            std::cout << "process_lock: ";
            for (auto pid: process_lock)
                std::cout << pid;
            std::cout << std::endl;

            std::cout << "candidate_pids: ";
            for (auto pid: candidate_pids) 
                std::cout << pid << " ";
            std::cout << std::endl;
            */



            std::vector<std::pair<int, int>> pid_tids = 
                select_threads_from_subset(instance, available_workers_, num_processes_, instance->get_num_workers(), candidate_pids);

            if (!pid_tids.empty()) {
                int j = 0;
                for (auto pid_tid : pid_tids) {
                    instance->add_thread(pid_tid.first, pid_tid.second, j++);
                }
                task_manager_.record_and_track(id, pid_tids);
                // erase this task from angry list
                // task_manager_.suc_sched(id);
                success_tasks.push_back(id);
                instances.push_back(std::move(instance));
                husky::LOG_I << YELLOW("Angry task: "+std::to_string(id)+" added");
            } else {
                // task_manager_.fail_sched(id);
                for (auto& pid : proc_ids) {
                    // lock those preferred by this angry task
                    process_lock.insert(pid); 
                }
            }
            begin++;
        }
        for (auto task_id : success_tasks) {
            task_manager_.suc_sched(task_id);
        }
        for (auto task_id : fail_tasks) {
            task_manager_.fail_sched(task_id);
        }
    }

    // 2. Handle tasks that are not angry and ready to run
    auto ordered_task = task_manager_.order_by_priority();
    for (auto& id : ordered_task) {
        std::shared_ptr<Instance> instance(new Instance);
        instance_basic_setup(instance, *(task_manager_.get_task_by_id(id)));
        std::vector<int> proc_ids = task_manager_.get_preferred_proc(id);

        std::vector<int> candidate_pids;
        // filter out locked processes
        for (auto& pid : proc_ids) {
            if (process_lock.find(pid) == process_lock.end()) {
                candidate_pids.push_back(pid);
            }
        }

        std::vector<std::pair<int, int>> pid_tids = 
            select_threads_from_subset(instance, available_workers_, num_processes_, instance->get_num_workers(), candidate_pids);

        if (!pid_tids.empty()) {
            int j = 0;
            for (auto pid_tid : pid_tids) {
                instance->add_thread(pid_tid.first, pid_tid.second, j++);
            }
            task_manager_.record_and_track(id, pid_tids);
            task_manager_.suc_sched(id);
            instances.push_back(std::move(instance));
            husky::LOG_I << YELLOW("Non-angry task: "+std::to_string(id)+" added");
        } else {
            task_manager_.fail_sched(id);
        }
    }
    return instances;
}

bool PriorityTaskScheduler::is_finished() {return task_manager_.is_finished(); }

} // namespace husky
