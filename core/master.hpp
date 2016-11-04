#pragma once

#include "base/debug.hpp"

#include "base/serialization.hpp"
#include "core/constants.hpp"
#include "core/worker_info.hpp"
#include "core/master_connection.hpp"
#include "core/workers_pool.hpp"
#include "core/assigner.hpp"
#include "core/instance.hpp"

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
        
        for (auto& socket : master_connection.get_sockets()) {
            std::cout << "[Master]: Trying to send to process "+std::to_string(socket.first) << std::endl;
            zmq_sendmore_int32(&socket.second, constants::TASK_TYPE);
            zmq_sendmore_string(&socket.second, "hello");
            std::cout << "[Master]: Send done" << std::endl;
            zmq_send_binstream(&socket.second, bin);
        }
    }

private:
    // Store the worker info
    WorkerInfo worker_info;

    // Store the available workers
    WorkersPool workers_pool;

    // connect to workers
    MasterConnection master_connection;

    // 
    Assigner assigner;
};

}  // namespace husky
