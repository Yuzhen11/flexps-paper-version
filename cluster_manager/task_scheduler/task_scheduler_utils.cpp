#include "cluster_manager/task_scheduler/task_scheduler_utils.hpp"

#include <boost/algorithm/string.hpp>
#include <iostream>

#include "cluster_manager/task_scheduler/available_workers.hpp"
#include "core/instance.hpp"
#include "core/task.hpp"

namespace husky {

void instance_basic_setup(std::shared_ptr<Instance>& instance, const Task& task) {
    // TODO If the task type is MLTaskType and the running type is unset,
    // need to decide it's real running type now
    if (task.get_type() == Task::Type::MLTaskType) {
        // TODO now set to SingleTaskType for testing...
        instance->set_task(task, "");
        // instance->set_task(task, Task::Type::HogwildTaskType);
    } else {
        instance->set_task(task);
    }

    // TODO: ClusterManager needs to design workers number for MLTaskType if user didn't set it
    if (task.get_type() == Task::Type::MLTaskType && task.get_num_workers() == 0)
        instance->set_num_workers(1);
}

std::vector<std::pair<int, int>> select_threads(std::shared_ptr<Instance>& instance, AvailableWorkers& available_workers, int num_processes) {
    // randomly select threads
    std::vector<std::pair<int, int>> pid_tids;
    if (instance->get_type() == Task::Type::ConfigurableWorkersTaskType) {
        std::vector<int> worker_num = static_cast<const ConfigurableWorkersTask*>(instance->get_task())->get_worker_num();
        std::vector<std::string> worker_num_type = static_cast<const ConfigurableWorkersTask*>(instance->get_task())->get_worker_num_type();
        int current_epoch = instance->get_epoch();
        int current_pos = current_epoch % worker_num.size();

        if (worker_num_type[current_pos] == "threads_per_worker") {
	       pid_tids = available_workers.get_workers_per_process(worker_num[current_pos], num_processes);
        }
        else if (worker_num_type[current_pos] == "threads_per_cluster") {
	       pid_tids = available_workers.get_workers(worker_num[current_pos]);
        }
        else if (worker_num_type[current_pos] == "local_threads") {
	       pid_tids = available_workers.get_local_workers(worker_num[current_pos]);
        } else if (worker_num_type[current_pos] == "threads_traverse_cluster") {
            pid_tids = available_workers.get_traverse_workers(instance->get_id(), worker_num[current_pos], num_processes); 
        } else if (worker_num_type[current_pos].find("threads_on_worker") != std::string::npos){
            std::vector<std::string> split_result;
            boost::split(split_result, worker_num_type[current_pos], boost::is_any_of(":"));
            if (split_result.size() == 2) {
                int pos_worker = std::stoi(split_result[1]);
                pid_tids = available_workers.get_workers_exact_process(worker_num[current_pos], pos_worker, num_processes);
            } else {
                husky::LOG_I << "illegal threads_on_worker of worker_num_type!";
            }
        }
        else {
	       husky::LOG_I << "illegal worker_num_type!";
        }
    } else if (instance->get_type() == Task::Type::MLTaskType) {
        auto& hint = instance->get_task()->get_hint();
        if (hint.at(husky::constants::kType) == husky::constants::kHogwild
            || hint.at(husky::constants::kType) == husky::constants::kSPMT) {
            // extract from one process
            pid_tids = available_workers.get_local_workers(instance->get_num_workers());
        } else {
            // extract from global workers
            pid_tids = available_workers.get_workers(instance->get_num_workers());
        }
    }
    else {
        // extract from global workers
        pid_tids = available_workers.get_workers(instance->get_num_workers());
    }

    return pid_tids;
}

std::vector<std::pair<int, int>> select_threads_from_subset(
        std::shared_ptr<Instance>& instance, AvailableWorkers& available_workers, int num_processes, 
        int required_num_threads, const std::vector<int>& candidate_proc) {

    std::vector<std::pair<int, int>> pid_tids;
    //TODO: schedule for other task type
    if (instance->get_type() == Task::Type::MLTaskType) {
        auto& hint = instance->get_task()->get_hint();
        if (hint.at(husky::constants::kType) == husky::constants::kSingle
            || hint.at(husky::constants::kType) == husky::constants::kHogwild
            || hint.at(husky::constants::kType) == husky::constants::kSPMT) {
            for (auto &pid : candidate_proc) {
                pid_tids = available_workers.get_workers_exact_process(required_num_threads, pid, num_processes);
                if (pid_tids.size() == required_num_threads) {
                    break;
                }
            }
            assert(pid_tids.size() == required_num_threads || pid_tids.size() == 0);
        } else { // PS:BSP SSP ASP
            // min_per_proc = min(available workers of all proc, average required threads for each proc)
            int min_per_proc = required_num_threads/num_processes;
            int sum_available = 0;
            for (auto pid : candidate_proc) {
                int num_available = available_workers.get_num_available_local_workers(pid);
                sum_available+= num_available;
                if (num_available < min_per_proc)
                    min_per_proc = num_available; 
            }
            
            if (sum_available >= required_num_threads) {
                // select min_per_proc threads from all candidate_proc
                if (min_per_proc>0) {
                    for (auto pid : candidate_proc) {
                        auto tmp_pid_tids = available_workers.get_workers_exact_process(min_per_proc, pid, num_processes);
                        for (auto& pid_tid : tmp_pid_tids) {
                            pid_tids.push_back(pid_tid);
                        }
                    }
                }
                // incrementally select threads from all proc 
                bool keep_looping = true;
                while (keep_looping) {
                    for (auto pid : candidate_proc) {
                        int num_available = available_workers.get_num_available_local_workers(pid);
                        if (num_available>0) {
                            auto tmp_pid_tids = available_workers.get_workers_exact_process(1, pid, num_processes);
                            pid_tids.push_back(tmp_pid_tids[0]);
                            if (pid_tids.size() == required_num_threads) {
                                keep_looping = false;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    if (!pid_tids.empty()) {
        std::cout<<"task "<<instance->get_id()<<" selcted:\n";
        for (auto& pid_tid : pid_tids) {
            std::cout<<pid_tid.first<<" "<<pid_tid.second<<std::endl;
        }
    }
    return pid_tids;
}

}  // namespace husky
