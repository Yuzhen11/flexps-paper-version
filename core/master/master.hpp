#pragma once

#include "base/debug.hpp"
#include "base/log.hpp"

#include "base/serialization.hpp"
#include "core/common/constants.hpp"
#include "core/common/instance.hpp"
#include "core/common/worker_info.hpp"
#include "core/master/master_connection.hpp"
#include "core/master/cluster_manager.hpp"

namespace husky {

class Master {
public:
    Master() = delete;
    Master(WorkerInfo&& worker_info_, MasterConnection&& master_connection_)
        :worker_info(std::move(worker_info_)),
        master_connection(std::move(master_connection_)),
        cluster_manager(worker_info, master_connection)
    {}

    void test_connection() {
        Instance instance(0);
        std::unordered_map<int, std::vector<int>> cluster;
        cluster.insert({0, {0,1}}); // {0, {0,1}}
        instance.set_cluster(cluster);  
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

    void recv_tasks_from_worker() {
        // recv tasks from proc 0 
        auto& socket = master_connection.get_recv_socket();
        auto bin = zmq_recv_binstream(&socket);
        std::vector<Task> tasks;
        size_t num_tasks;
        bin >> num_tasks;
        for (int i = 0; i < num_tasks; ++ i) {
            Task task;
            bin >> task;
            tasks.push_back(std::move(task));
            cluster_manager.init_tasks(tasks);
        }
        base::log_msg("[Master]: recv_task_from_wroker done");
        base::log_msg("[Master]: Totally "+std::to_string(num_tasks)+" tasks received");
    }

    void assign_initial_tasks() {
        cluster_manager.assign_next_tasks();
    }

    void master_loop() {
        auto& recv_socket = master_connection.get_recv_socket();
        while (true) {
            auto bin = zmq_recv_binstream(&recv_socket);
            cluster_manager.finish_local_instance(bin);
            cluster_manager.assign_next_tasks();

            bool is_finished = cluster_manager.is_finished();
            if (is_finished) {
                break;
            }
        }
    }


private:
    // store the worker info
    WorkerInfo worker_info;

    // connect to workers
    MasterConnection master_connection;

    // manage the cluster
    ClusterManager cluster_manager;
};

}  // namespace husky
