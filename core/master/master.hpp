#pragma once

#include "base/debug.hpp"
#include "base/log.hpp"

#include "base/serialization.hpp"
#include "core/common/constants.hpp"
#include "core/common/instance.hpp"
#include "core/common/worker_info.hpp"
#include "core/master/master_connection.hpp"
#include "core/master/workers_pool.hpp"
#include "core/master/task_manager.hpp"

namespace husky {

class Master {
public:
    Master() = delete;
    Master(WorkerInfo&& worker_info_, WorkersPool&& workers_pool_, MasterConnection&& master_connection_)
        :worker_info(std::move(worker_info_)),
        workers_pool(std::move(workers_pool_)),
        master_connection(std::move(master_connection_)) 
    {}

    void test_connection() {
        Cluster cluster({0,1});
        Instance instance(0, cluster);
        base::BinStream bin;
        bin << instance;
        
        for (auto& socket : master_connection.get_send_sockets()) {
            base::log_msg("[Master]: Trying to send to process "+std::to_string(socket.first));
            zmq_sendmore_int32(&socket.second, constants::TASK_TYPE);
            zmq_sendmore_string(&socket.second, "hello");
            base::log_msg("[Master]: Send done");
            zmq_send_binstream(&socket.second, bin);
        }
    }

    void init_tasks() {
        // recv tasks from proc 0 
        auto& socket = master_connection.get_recv_socket();
        auto bin = zmq_recv_binstream(&socket);
        int num_tasks;
        bin >> num_tasks;
        for (int i = 0; i < num_tasks; ++ i) {
            Task task;
            bin >> task;
            task_manager.add_task(task);
        }
        base::log_msg("[Master]: Init tasks done");
    }
    void master_loop() {
    }


private:
    // Store the worker info
    WorkerInfo worker_info;

    // Store the available workers
    WorkersPool workers_pool;

    // connect to workers
    MasterConnection master_connection;

    // store the tasks
    TaskManager task_manager;
};

}  // namespace husky
