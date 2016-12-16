#pragma once

#include "husky/base/exception.hpp"
#include "husky/base/log.hpp"
#include "husky/core/worker_info.hpp"

#include "core/constants.hpp"
#include "core/instance.hpp"
#include "husky/base/serialization.hpp"
#include "cluster_manager/cluster_manager_connection.hpp"
#include "cluster_manager/task_scheduler.hpp"

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
    ClusterManager() = delete;
    ClusterManager(WorkerInfo&& worker_info_, ClusterManagerConnection&& cluster_manager_connection_)
        : worker_info(std::move(worker_info_)),
          cluster_manager_connection(std::move(cluster_manager_connection_)),
          task_scheduler(new SequentialTaskScheduler(worker_info)) {}

    /*
     * The main loop for cluster_manager logic
     */
    void cluster_manager_loop() {
        auto& recv_socket = cluster_manager_connection.get_recv_socket();
        while (true) {
            int type = zmq_recv_int32(&recv_socket);
            base::log_msg("[ClusterManager]: Type: " + std::to_string(type));
            if (type == constants::kClusterManagerInit) {
                // 1. Received tasks from Worker
                recv_tasks_from_worker();

                // 2. Extract instances
                extract_instaces();
            } else if (type == constants::kClusterManagerInstanceFinished) {
                // 1. Receive finished instances
                auto bin = zmq_recv_binstream(&recv_socket);
                int instance_id, proc_id;
                bin >> instance_id >> proc_id;
                task_scheduler->finish_local_instance(instance_id, proc_id);
                base::log_msg("[ClusterManager]: task id: " + std::to_string(instance_id) + " proc id: " +
                              std::to_string(proc_id) + " done");

                // 2. Extract instances
                extract_instaces();
            } else if (type == constants::kClusterManagerExit) {
                break;
            } else {
                throw base::HuskyException("[ClusterManager] ClusterManager Loop recv type error, type is: " + std::to_string(type));
            }
        }
    }

   private:
    /*
     * Recv tasks from worker and pass the tasks to TaskScheduler
     */
    void recv_tasks_from_worker() {
        // recv tasks from proc 0
        auto& socket = cluster_manager_connection.get_recv_socket();
        auto bin = zmq_recv_binstream(&socket);
        auto tasks = task::extract_tasks(bin);
        task_scheduler->init_tasks(tasks);
        for (auto& task : tasks) {
            base::log_msg("[ClusterManager]: Task: " + std::to_string(task->get_id()) + " added");
        }
        base::log_msg("[ClusterManager]: Totally " + std::to_string(tasks.size()) + " tasks received");
    }

    /*
     * Extract instances and assign them to Workers
     * If no more instances, send exit signal
     */
    void extract_instaces() {
        // 1. Extract and assign next instances
        auto instances = task_scheduler->extract_instances();
        send_instances(instances);

        // 2. Check whether all tasks have finished
        bool is_finished = task_scheduler->is_finished();
        if (is_finished) {
            send_exit_signal();
        }
    }

    /*
     * Send instances to Workers
     */
    void send_instances(const std::vector<std::shared_ptr<Instance>>& instances) {
        base::log_msg("[ClusterManager]: Assigning next instances");
        auto& sockets = cluster_manager_connection.get_send_sockets();
        for (auto& instance : instances) {
            instance->show_instance();
            base::BinStream bin;
            // TODO Support different types of instance in hierarchy
            instance->serialize(bin);
            auto& cluster = instance->get_cluster();
            for (auto& kv : cluster) {
                auto it = sockets.find(kv.first);
                zmq_sendmore_int32(&it->second, constants::kTaskType);
                zmq_send_binstream(&it->second, bin);
            }
        }
    }
    /*
     * Send exit signal to Workers
     */
    void send_exit_signal() {
        auto& proc_sockets = cluster_manager_connection.get_send_sockets();
        for (auto& socket : proc_sockets) {
            zmq_send_int32(&socket.second, constants::kClusterManagerFinished);
        }
    }

   private:
    // store the worker info
    WorkerInfo worker_info;

    // connect to workers
    ClusterManagerConnection cluster_manager_connection;

    // task scheduler
    std::unique_ptr<TaskScheduler> task_scheduler;
};

}  // namespace husky
