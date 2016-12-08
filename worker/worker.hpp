#pragma once

#include <type_traits>

#include "husky/base/log.hpp"
#include "husky/base/exception.hpp"
#include "husky/base/serialization.hpp"
#include "husky/core/zmq_helpers.hpp"
#include "base/debug.hpp"
#include "core/constants.hpp"
#include "core/instance.hpp"
#include "core/worker_info.hpp"
#include "worker/master_connector.hpp"
#include "worker/instance_runner.hpp"
#include "worker/task_store.hpp"
#include "worker/basic.hpp"

namespace husky {

/*
 * Worker contains the event-loop to receive tasks from Master,
 * and also has function to communicate with Master, 
 * like send_tasks_to_master(), send_exit()
 */
class Worker {
public:
    Worker() = delete;
    Worker(WorkerInfo&& worker_info_, MasterConnector&& master_connector_)
        : worker_info(std::move(worker_info_)),
        master_connector(std::move(master_connector_)),
        instance_runner(worker_info, master_connector, task_store){
    }

    // User need to add task to taskstore
    void add_task(std::unique_ptr<Task>&& task, const FuncT& func) {
        task_store.add_task(std::move(task), func);
    }

    void send_tasks_to_master() {
        // Only Proc 0 need to send tasks to master
        if (worker_info.get_process_id() == 0) {
            base::BinStream bin;
            auto& task_map = task_store.get_task_map();
            auto& buffered_tasks = task_store.get_buffered_tasks();
            // send out buffered_tasks
            bin << buffered_tasks.size();
            for (auto id : buffered_tasks) {
                auto& task = task_map[id].first;
                bin << task->get_type();  // push the task type first
                task->serialize(bin);  // push the task
            }
            auto& socket = master_connector.get_send_socket();
            zmq_sendmore_int32(&socket, constants::MASTER_INIT);
            zmq_send_binstream(&socket, bin);
            base::log_msg("[Worker]: Totally "+std::to_string(buffered_tasks.size())+" tasks sent");
            // clear buffered tasks
            task_store.clear_buffered_tasks();
        }
    }

    /*
     * send exit signal to master, stop the master
     * normally it's the last statement in worker
     */
    void send_exit() {
        if (worker_info.get_process_id() == 0) {
            auto& socket = master_connector.get_send_socket();
            zmq_send_int32(&socket, constants::MASTER_EXIT);
        }
    }

    void send_instance_finished(base::BinStream& bin) {
        auto& socket = master_connector.get_send_socket();
        zmq_sendmore_int32(&socket, constants::MASTER_INSTANCE_FINISHED);
        zmq_send_binstream(&socket, bin);  // {instance_id, proc_id}
    }

    void main_loop() {
        auto& socket = master_connector.get_recv_socket();
        auto& send_socket = master_connector.get_send_socket();
        while (true) {
            int type = zmq_recv_int32(&socket);
            // base::log_msg("[Worker]: Msg Type: " + std::to_string(type));
            if (type == constants::TASK_TYPE) {
                auto bin = zmq_recv_binstream(&socket);
                // TODO Support different types of instance in hierarchy
                std::shared_ptr<Instance> instance(new Instance);
                instance->deserialize(bin);
                // Print debug info
                instance->show_instance(worker_info.get_process_id());
                instance_runner.run_instance(instance);
            }
            else if (type == constants::THREAD_FINISHED) {
                int instance_id = zmq_recv_int32(&socket);
                int thread_id = zmq_recv_int32(&socket);
                instance_runner.finish_thread(instance_id, thread_id);
                bool is_instance_done = instance_runner.is_instance_done(instance_id);
                if (is_instance_done) {
                    base::log_msg("[Worker]: task id:"+std::to_string(instance_id)+" finished on Proc:"+std::to_string(worker_info.get_process_id()));
                    auto bin = instance_runner.remove_instance(instance_id);
                    send_instance_finished(bin);
                }
            }
            else if (type == constants::MASTER_FINISHED) {
                base::log_msg("[Worker]: worker exit");
                break;
            }
            else {
                throw base::HuskyException("[Worker] Worker Loop recv type error, type is: "+std::to_string(type));
            }
        }
    }

private:
    MasterConnector master_connector;
    WorkerInfo worker_info;
    InstanceRunner instance_runner;
    TaskStore task_store;
};

}  // namespace husky
