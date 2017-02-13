#pragma once

#include "husky/base/exception.hpp"
#include "husky/base/log.hpp"
#include "husky/core/worker_info.hpp"

#include "cluster_manager/cluster_manager_connection.hpp"
#include "cluster_manager/task_scheduler/greedy_task_scheduler.hpp"
#include "cluster_manager/task_scheduler/sequential_task_scheduler.hpp"
#include "cluster_manager/task_scheduler/history_manager.hpp"
#include "core/constants.hpp"
#include "core/instance.hpp"
#include "husky/base/serialization.hpp"

namespace husky {

/*
 * ClusterManager implements the ClusterManager logic
 *
 * Basically it runs an event-loop to receive signal from Workers and react accordingly.
 *
 * The event may be kClusterManagerInit, kClusterManagerInstanceFinished, kClusterManagerExit.
 */
class ClusterManager {
   public:
    ClusterManager() = default;

    // setup
    void setup(WorkerInfo&& worker_info, ClusterManagerConnection&& cluster_manager_connection,
               const std::string& hint = "") {
        worker_info_ = std::move(worker_info);
        cluster_manager_connection_.reset(new ClusterManagerConnection(std::move(cluster_manager_connection)));
        setup_task_scheduler(hint);
        // init history manager map
        HistoryManager::get().start(worker_info_.get_num_processes());
        is_setup = true;
    }

    void setup_task_scheduler(const std::string& hint) {
        if (hint == "greedy") {
            task_scheduler_.reset(new GreedyTaskScheduler(worker_info_));
            husky::LOG_I << "[ClusterManager]: TaskScheduler set to Greedy";
        } else if (hint == "sequential") {
            task_scheduler_.reset(new SequentialTaskScheduler(worker_info_));
            husky::LOG_I << "[ClusterManager]: TaskScheduler set to Sequential";
        } else if (hint == "") {  // The default is sequential
            task_scheduler_.reset(new SequentialTaskScheduler(worker_info_));
            husky::LOG_I << "[ClusterManager]: TaskScheduler set to Sequential";
        } else {
            throw base::HuskyException("[ClusterManager] setup_task_scheduler failed, unknown hint: " + hint);
        }
    }

    /*
     * The main loop for cluster_manager logic
     */
    void serve() {
        assert(is_setup);
        auto& recv_socket = cluster_manager_connection_->get_recv_socket();
        while (true) {
            int type = zmq_recv_int32(&recv_socket);
            // husky::LOG_I << "[ClusterManager]: Type: " + std::to_string(type);
            if (type == constants::kClusterManagerInit) {
                // 1. Received tasks from Worker
                recv_tasks_from_worker();

                // 2. Extract instances
                extract_instaces();
            } else if (type == constants::kClusterManagerThreadFinished) {
                // 1. Receive finished thread
                auto bin = zmq_recv_binstream(&recv_socket);
                int instance_id, global_thread_id;
                bin >> instance_id >> global_thread_id;
                task_scheduler_->finish_thread(instance_id, global_thread_id);
                husky::LOG_I << CLAY("[ClusterManager]: task id: " + std::to_string(instance_id) + " thread id: " +
                                     std::to_string(global_thread_id) + " done");

                // 2. Extract instances
                extract_instaces();
            } else if (type == constants::kClusterManagerExit) {
                break;
            } else {
                throw base::HuskyException("[ClusterManager] ClusterManager Loop recv type error, type is: " +
                                           std::to_string(type));
            }
        }
    }

   private:
    /*
     * Recv tasks from worker and pass the tasks to TaskScheduler
     */
    void recv_tasks_from_worker() {
        // recv tasks from proc 0
        auto& socket = cluster_manager_connection_->get_recv_socket();
        auto bin = zmq_recv_binstream(&socket);
        auto tasks = task::extract_tasks(bin);
        task_scheduler_->init_tasks(tasks);
        // for (auto& task : tasks) {
        //     husky::LOG_I << "[ClusterManager]: Task: " + std::to_string(task->get_id()) + " added";
        // }
        husky::LOG_I << BLUE("[ClusterManager]: " + std::to_string(tasks.size()) + " tasks received");
    }

    /*
     * Extract instances and assign them to Workers
     * If no more instances, send exit signal
     */
    void extract_instaces() {
        // 1. Extract and assign next instances
        auto instances = task_scheduler_->extract_instances();
        if (!instances.empty())
            send_instances(instances);

        // 2. Check whether all tasks have finished
        bool is_finished = task_scheduler_->is_finished();
        if (is_finished) {
            send_exit_signal();
        }
    }

    /*
     * Send instances to Workers
     */
    void send_instances(const std::vector<std::shared_ptr<Instance>>& instances) {
        husky::LOG_I << "[ClusterManager]: Assigning next instances, size is " + std::to_string(instances.size());
        auto& sockets = cluster_manager_connection_->get_send_sockets();
        for (auto& instance : instances) {
            instance->show_instance();
            base::BinStream bin;
            // TODO Support different types of instance in hierarchy
            instance->serialize(bin);
            auto& proc_sockets = cluster_manager_connection_->get_send_sockets();
            for (auto& socket : proc_sockets) {  // send to all processes
                zmq_sendmore_int32(&socket.second, constants::kTaskType);
                zmq_send_binstream(&socket.second, bin);
            }
        }
    }
    /*
     * Send exit signal to Workers
     */
    void send_exit_signal() {
        auto& proc_sockets = cluster_manager_connection_->get_send_sockets();
        for (auto& socket : proc_sockets) {
            zmq_send_int32(&socket.second, constants::kClusterManagerFinished);
        }
    }

   private:
    // store the worker info
    WorkerInfo worker_info_;

    // connect to workers
    std::unique_ptr<ClusterManagerConnection> cluster_manager_connection_;

    // task scheduler
    std::unique_ptr<TaskScheduler> task_scheduler_;

    bool is_setup = false;
};

}  // namespace husky
