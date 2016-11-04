#pragma once

#include <thread>
#include "zmq.hpp" 

#include "base/debug.hpp"

#include "core/instance.hpp"
#include "core/worker_info.hpp"
#include "core/zmq_helpers.hpp"

namespace husky {

// Instances run on threads, InstanceManager keep track of the 
// instances and threads
class InstanceManager {
public:
    InstanceManager() = delete;
    InstanceManager(WorkerInfo& worker_info_, const std::string main_loop_port_, zmq::context_t& zmq_context_)
        : worker_info(worker_info_),
          main_loop_port(main_loop_port_),
          zmq_context(zmq_context_)
    {}

    Instance extract_local_instance(const Instance& instance) const {
        Instance local_instance(instance.get_id());
        for (auto tid : instance.get_threads()) {
            if (worker_info.get_proc_id(tid) == worker_info.get_proc_id()) {
                local_instance.get_cluster().add(worker_info.global_to_local_id(tid));
            }
        }
        return local_instance;
    }

    void run_instance(const Instance& instance) {
        assert(instances.find(instance) == instances.end());
        std::cout << "[Instance]: instance " + std::to_string(instance.get_id()) + " added" << std::endl;
        instances.insert(instance);


        std::string connect_addr = "tcp://"+worker_info.get_host()+":"+main_loop_port;
        int instance_id = instance.get_id();
        auto local_instance = extract_local_instance(instance);
        for (auto tid : local_instance.get_threads()) {
            // worker threads
            std::thread([this, instance_id, tid, connect_addr](){
                zmq::socket_t socket(zmq_context, ZMQ_PUSH);
                std::cout << "[Thread]: Hello World" << std::endl;
                std::cout << "[Thread]: " + connect_addr << std::endl;
                socket.connect(connect_addr);
                zmq_sendmore_int32(&socket, constants::THREAD_FINISHED);
                zmq_sendmore_int32(&socket, instance_id);
                zmq_send_int32(&socket, tid);
            }).detach();
        }
        std::unordered_set<int> s(local_instance.get_threads().begin(), local_instance.get_threads().end());
        instance_keeper.insert({instance.get_id(), std::move(s)});
    }

    void finailize_instance(int id) {
        std::cout << "[Instance]: instance " + std::to_string(id) + " finished" << std::endl;
        instances.erase(Instance(id));
    }

    void finish_thread(int instance_id, int tid) {
        instance_keeper[instance_id].erase(tid);
    }
    bool is_instance_done(int instance_id) {
        return instance_keeper[instance_id].empty();
    }
    void remove_instance(int instance_id) {
        assert(instance_keeper[instance_id].empty());
        instances.erase(Instance(instance_id));
        instance_keeper.erase(instance_id);
    }

private:
    std::string main_loop_port;
    zmq::context_t& zmq_context;
    WorkerInfo& worker_info;
    std::unordered_set<int> available_threads;
    std::unordered_set<Instance, InstanceHasher> instances;
    std::unordered_map<int, std::unordered_set<int>> instance_keeper;
};

}  // namespace husky
