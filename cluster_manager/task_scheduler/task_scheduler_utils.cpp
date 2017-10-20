#include "cluster_manager/task_scheduler/task_scheduler_utils.hpp"

#include <boost/algorithm/string.hpp>
#include <iostream>

#include "cluster_manager/task_scheduler/available_workers.hpp"
#include "cluster_manager/task_scheduler/history_manager.hpp"
#include "core/instance.hpp"
#include "core/task.hpp"
#include "worker/engine.hpp"

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

void global_guarantee_threads(const std::vector<std::shared_ptr<Task>>& tasks) {
    // worker info
    WorkerInfo worker_info = Context::get_worker_info();
    int total_workers = Context::get_num_workers();
    int total_process = Context::get_num_processes();
    bool is_guarantee = true;
    std::string error_msg;
    for(auto task:tasks) {
        if (task->get_type() == Task::Type::ConfigurableWorkersTaskType) {
            std::vector<int> worker_num = static_cast<const ConfigurableWorkersTask*>(task.get())->get_worker_num();
            std::vector<std::string> worker_num_type = static_cast<const ConfigurableWorkersTask*>(task.get())->get_worker_num_type();
            for(int i = 0; i < worker_num_type.size(); i++) {
                std::string type = worker_num_type[i];

                /* std::vector<std::string> worker_num_type_;
                 *
                 * worker_num_type equal "threads_per_worker", run 5 threads per worker
                 * worker_num_type equal "threads_per_cluster", run 5 threads per cluster
                 * worker_num_type equal "local_threads", run 5 local threads
                 * worker_num_type equal "threads_traverse_cluster", run 5 threads per worker by per worker
                 * worker_num_type equal "threads_on_worker:2", run 5 threads on worker 2
                 */
                if (type == "threads_per_worker") {
                    if (total_workers < worker_num[i] * total_process) {
                        is_guarantee = false;
                        error_msg = "task_id_" + std::to_string(task->get_id()) + " ConfigurableWorkersTask threads_per_worker exceed.";
                        break;
                    }
                } else if (type == "threads_per_cluster") {
                    if (total_workers < worker_num[i]) {
                        is_guarantee = false;
                        error_msg = "task_id_" + std::to_string(task->get_id()) + " ConfigurableWorkersTask threads_per_cluster exceed.";
                        break;
                    }
                } else if (type == "local_threads") {   // local_threads means making sure all seleted threads are in the same machine
                    // assume is_guarantee = false
                    // if none machine has enough local threads, assumption is right
                    is_guarantee = false;
                    std::vector<int> pids = worker_info.get_pids();
                    for(auto pid : pids) {
                        if (worker_info.get_num_local_workers(pid) >= worker_num[i]) {
                            is_guarantee = true;
                            break; 
                        }
                    }
                    if(!is_guarantee) {
                        error_msg = "task_id_" + std::to_string(task->get_id()) + " ConfigurableWorkersTask local_threads exceed.";
                        break;
                    }
                } else if (type == "threads_traverse_cluster") {
                    // get all pid
                    std::vector<int> pids = worker_info.get_pids();
                    for(auto pid : pids) {
                        if (worker_info.get_num_local_workers(pid) < worker_num[i]) {
                            is_guarantee = false;
                            error_msg = "task_id_" + std::to_string(task->get_id()) + " ConfigurableWorkersTask threads_traverse_cluster/pid_" + std::to_string(pid) + " exceed.";
                            break; 
                        }
                    }
                    if(!is_guarantee) {
                        break;
                    }
                } else if (type.find("threads_on_worker") != std::string::npos) {
                    std::vector<std::string> split_result;
                    boost::split(split_result, type, boost::is_any_of(":"));
                    if (split_result.size() == 2) {
                        int pos_worker = std::stoi(split_result[1]);
                        if (worker_info.get_num_local_workers(pos_worker) < worker_num[i]) {
                            is_guarantee = false;
                            error_msg = "task_id_" + std::to_string(task->get_id()) + " ConfigurableWorkersTask threads_on_worker/" + split_result[1] + " exceed.";
                            break; 
                        }
                    } else {
                       throw base::HuskyException("[task_scheduler_utils] illegal threads_on_worker of worker_num about ConfigurableWorkersTask.");
                    }
                } else {
                    throw base::HuskyException("[task_scheduler_utils] illegal threads_on_worker of worker_num_type about ConfigurableWorkersTask.");
                }
            }

            if (!is_guarantee) {
                break;
            }
        } else {
            if (total_workers < task->get_num_workers()){
                is_guarantee = false;
                error_msg = "task_id_" + std::to_string(task->get_id()) + " exceed.";
                break;
            }
        }
    }

    if (!is_guarantee) {
        throw base::HuskyException("[task_scheduler_utils] max num of threads overflow. Details: " + error_msg);
    }
}

// Select the processes which are least visited by this task as candidates
std::vector<int> get_preferred_proc(int task_id) {
    std::vector<int> task_history = HistoryManager::get().get_task_history(task_id);
    std::vector<int> plan;
    int smallest = std::numeric_limits<int>::max();
    for (int i=0; i<task_history.size(); i++) {
        if (task_history[i] < smallest) {
            smallest = task_history[i];
        }
    }
    for (int i=0; i<task_history.size(); i++) {
        if (task_history[i] == smallest) {
            plan.push_back(i);
        }
    }
    return plan;
}

std::vector<std::pair<int, int>> select_threads_from_subset(
        std::shared_ptr<Instance>& instance, AvailableWorkers& available_workers, int num_processes, 
        int required_num_threads, const std::vector<int>& candidate_proc) {

    std::vector<std::pair<int, int>> pid_tids;
    auto& hint = instance->get_task()->get_hint();
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
    } else if (instance->get_type() == Task::Type::MLTaskType && (
            hint == husky::constants::kSingle
            || hint == husky::constants::kHogwild
            || hint == husky::constants::kSPMT)
                ) {  // if task is MLTask and kType is kSingle/kHogwild/kSPMT
        if (hint == husky::constants::kSingle) {  // Single must use 1 thread
            assert(required_num_threads == 1);
        }
        for (auto &pid : candidate_proc) {
            pid_tids = available_workers.get_workers_exact_process(required_num_threads, pid, num_processes);
            if (pid_tids.size() == required_num_threads) {
                break;
            }
        }
        assert(pid_tids.size() == required_num_threads || pid_tids.size() == 0);
    } else { // Other types of job or PS
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
            if (pid_tids.size() != required_num_threads) {  // If not all threads are found
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

    return pid_tids;
}

}  // namespace husky
