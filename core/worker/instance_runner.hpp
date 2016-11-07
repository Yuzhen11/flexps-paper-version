#pragma once

#include <thread>
#include "zmq.hpp" 

#include "base/debug.hpp"
#include "base/log.hpp"
#include "base/serialization.hpp"

#include "core/common/instance.hpp"
#include "core/common/worker_info.hpp"
#include "core/common/zmq_helpers.hpp"
#include "core/worker/master_connector.hpp"

namespace husky {

// Instances run on threads, InstanceRunner keep track of the 
// instances and threads
class InstanceRunner {
public:
    InstanceRunner() = delete;
    InstanceRunner(WorkerInfo& worker_info_, MasterConnector& master_connector_)
        : worker_info(worker_info_),
        master_connector(master_connector_)
    {}

    std::vector<int> extract_local_instance(const Instance& instance) const {
        std::vector<int> local_threads;
        local_threads = instance.get_threads(worker_info.get_proc_id());
        for (auto& th : local_threads) {
            th = worker_info.global_to_local_id(th);
        }
        return local_threads;
    }

    void run_instance(const Instance& instance) {
        assert(instances.find(instance.get_id()) == instances.end());
        auto instance_id = instance.get_id();
        // retrieve local threads
        auto local_threads = extract_local_instance(instance);
        instances.insert({instance.get_id(), instance});  // store the instance

        for (auto tid : local_threads) {
            // worker threads
            std::thread([this, instance_id, tid](){
                zmq::socket_t socket = master_connector.get_socket_to_recv();
                base::log_msg("[Thread]: Hello World");
                // tell worker when I finished
                zmq_sendmore_int32(&socket, constants::THREAD_FINISHED);
                zmq_sendmore_int32(&socket, instance_id);
                zmq_send_int32(&socket, tid);
            }).detach();
        }
        std::unordered_set<int> local_threads_set(local_threads.begin(), local_threads.end());
        instance_keeper.insert({instance_id, std::move(local_threads_set)});
        base::log_msg("[InstanceRunner]: instance " + std::to_string(instance.get_id()) + " added");
    }

    void finish_thread(int instance_id, int tid) {
        instance_keeper[instance_id].erase(tid);
        base::log_msg("[InstanceRunner]: instance_id: " + std::to_string(instance_id) + " tid: " + std::to_string(tid) + " finished");
    }
    bool is_instance_done(int instance_id) {
        return instance_keeper[instance_id].empty();
    }
    void remove_instance(int instance_id) {
        assert(instance_keeper[instance_id].empty());
        instances.erase(instance_id);
        instance_keeper.erase(instance_id);

        // tell master
        auto proc_id = worker_info.get_proc_id();
        base::BinStream bin;
        bin << instance_id;
        bin << proc_id;
        auto& socket = master_connector.get_send_socket();
        zmq_send_binstream(&socket, bin);  // {instance_id, proc_id}

        base::log_msg("[InstanceRunner]: instance " + std::to_string(instance_id) + " done ");
    }

private:
    WorkerInfo& worker_info;
    MasterConnector& master_connector;
    std::unordered_map<int, Instance> instances;
    std::unordered_map<int, std::unordered_set<int>> instance_keeper;
};

}  // namespace husky
