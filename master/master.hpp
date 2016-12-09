#pragma once

#include "husky/base/log.hpp"
#include "husky/base/exception.hpp"
#include "husky/core/worker_info.hpp"

#include "husky/base/serialization.hpp"
#include "core/constants.hpp"
#include "core/instance.hpp"
#include "master/master_connection.hpp"
#include "master/task_scheduler.hpp"

namespace husky {

/*
 * Master implements the Master logic
 *
 * Basically it runs an event-loop to receive signal from Workers and react accordingly.
 *
 * The event may be kMasterInit, kMasterInstanceFinished, kMasterExit.
 */
class Master {
public:
    Master() = delete;
    Master(WorkerInfo&& worker_info_, MasterConnection&& master_connection_)
        :worker_info(std::move(worker_info_)),
        master_connection(std::move(master_connection_)),
        task_scheduler(new SequentialTaskScheduler(worker_info))
    {}

    /* 
     * The main loop for master logic
     */
    void master_loop() {
        auto& recv_socket = master_connection.get_recv_socket();
        while (true) {
            int type = zmq_recv_int32(&recv_socket);
            base::log_msg("[Master]: Type: "+std::to_string(type));
            if (type == constants::kMasterInit) {
                // 1. Received tasks from Worker
                recv_tasks_from_worker();

                // 2. Extract instances
                extract_instaces();
            }
            else if (type == constants::kMasterInstanceFinished) {
                // 1. Receive finished instances
                auto bin = zmq_recv_binstream(&recv_socket);
                int instance_id, proc_id;
                bin >> instance_id >> proc_id;
                task_scheduler->finish_local_instance(instance_id, proc_id);
                base::log_msg("[Master]: task id: "+std::to_string(instance_id)+" proc id: "+std::to_string(proc_id) + " done");

                // 2. Extract instances
                extract_instaces();
            }
            else if (type == constants::kMasterExit) {
                break;
            }
            else {
                throw base::HuskyException("[Master] Master Loop recv type error, type is: "+std::to_string(type));
            }
        }
    }


private:
    /*
     * Recv tasks from worker and pass the tasks to TaskScheduler
     */
    void recv_tasks_from_worker() {
        // recv tasks from proc 0 
        auto& socket = master_connection.get_recv_socket();
        auto bin = zmq_recv_binstream(&socket);
        auto tasks = task::extract_tasks(bin);
        task_scheduler->init_tasks(tasks);
        for (auto& task : tasks) {
            base::log_msg("[Master]: Task: "+std::to_string(task->get_id())+" added");
        }
        base::log_msg("[Master]: Totally "+std::to_string(tasks.size())+" tasks received");
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
        base::log_msg("[Master]: Assigning next instances");
        auto& sockets = master_connection.get_send_sockets();
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
        auto& proc_sockets = master_connection.get_send_sockets();
        for (auto& socket : proc_sockets) {
            zmq_send_int32(&socket.second, constants::kMasterFinished);
        }
    }

private:
    // store the worker info
    WorkerInfo worker_info;

    // connect to workers
    MasterConnection master_connection;

    // task scheduler
    std::unique_ptr<TaskScheduler> task_scheduler;
};

}  // namespace husky
