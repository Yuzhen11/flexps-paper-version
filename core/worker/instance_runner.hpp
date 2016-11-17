#pragma once

#include <thread>
#include "zmq.hpp" 

#include "base/debug.hpp"
#include "base/log.hpp"
#include "base/serialization.hpp"

#include "core/common/instance.hpp"
#include "core/common/worker_info.hpp"
#include "core/common/zmq_helpers.hpp"
#include "core/common/info.hpp"
#include "core/common/utility.hpp"
#include "core/worker/master_connector.hpp"
#include "core/worker/task_store.hpp"

namespace husky {

// Instances run on threads, InstanceRunner keep track of the 
// instances and threads
class InstanceRunner {
public:
    InstanceRunner() = delete;
    InstanceRunner(WorkerInfo& worker_info_, MasterConnector& master_connector_, TaskStore& task_store_)
        : worker_info(worker_info_),
        master_connector(master_connector_),
        task_store(task_store_)
    {}

    std::vector<std::pair<int,int>> extract_local_instance(const Instance& instance) const {
        auto local_threads = instance.get_threads(worker_info.get_proc_id());
        for (auto& th : local_threads) {
            th.first = worker_info.global_to_local_id(th.first);
        }
        return local_threads;
    }

    void run_instance(const Instance& instance) {
        assert(instances.find(instance.get_id()) == instances.end());
        // retrieve local threads
        auto local_threads = extract_local_instance(instance);
        instances.insert({instance.get_id(), instance});  // store the instance

        for (auto tid : local_threads) {
            // worker threads
            std::thread([this, instance, tid](){
                zmq::socket_t socket = master_connector.get_socket_to_recv();
                // set the info
                Info info = utility::instance_to_info(instance);
                info.local_id = tid.first;
                info.global_id = worker_info.local_to_global_id(tid.first);
                info.cluster_id = tid.second;
                info.num_local_threads = instance.get_threads(worker_info.get_proc_id()).size();
                info.num_global_threads = instance.get_num_threads();
                // run the UDF!!!
                task_store.get_func(instance.get_id())(info);
                // tell worker when I finished
                zmq_sendmore_int32(&socket, constants::THREAD_FINISHED);
                zmq_sendmore_int32(&socket, instance.get_id());
                zmq_send_int32(&socket, tid.first);
            }).detach();
        }
        std::unordered_set<int> local_threads_set;
        for (auto tid : local_threads)
            local_threads_set.insert(tid.first);
        instance_keeper.insert({instance.get_id(), std::move(local_threads_set)});
        // base::log_msg("[InstanceRunner]: instance " + std::to_string(instance.get_id()) + " added");
    }

    void finish_thread(int instance_id, int tid) {
        instance_keeper[instance_id].erase(tid);
        // base::log_msg("[InstanceRunner]: instance_id: " + std::to_string(instance_id) + " tid: " + std::to_string(tid) + " finished");
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
    }

private:
    WorkerInfo& worker_info;
    MasterConnector& master_connector;
    TaskStore& task_store;
    std::unordered_map<int, Instance> instances;
    std::unordered_map<int, std::unordered_set<int>> instance_keeper;
};

}  // namespace husky
