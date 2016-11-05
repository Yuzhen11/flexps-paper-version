#pragma once

#include <thread>
#include "zmq.hpp" 

#include "base/debug.hpp"
#include "base/log.hpp"

#include "core/common/instance.hpp"
#include "core/common/worker_info.hpp"
#include "core/common/zmq_helpers.hpp"
#include "core/worker/master_connector.hpp"

namespace husky {

// Instances run on threads, InstanceManager keep track of the 
// instances and threads
class InstanceManager {
public:
    InstanceManager() = delete;
    InstanceManager(WorkerInfo& worker_info_, MasterConnector& master_connector_)
        : worker_info(worker_info_),
        master_connector(master_connector_)
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
        base::log_msg("[Instance]: instance " + std::to_string(instance.get_id()) + " added");
        instances.insert(instance);

        int instance_id = instance.get_id();
        auto local_instance = extract_local_instance(instance);
        for (auto tid : local_instance.get_threads()) {
            // worker threads
            std::thread([this, instance_id, tid](){
                zmq::socket_t socket = master_connector.get_socket_to_recv();
                base::log_msg("[Thread]: Hello World");
                zmq_sendmore_int32(&socket, constants::THREAD_FINISHED);
                zmq_sendmore_int32(&socket, instance_id);
                zmq_send_int32(&socket, tid);
            }).detach();
        }
        std::unordered_set<int> s(local_instance.get_threads().begin(), local_instance.get_threads().end());
        instance_keeper.insert({instance.get_id(), std::move(s)});
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
    WorkerInfo& worker_info;
    MasterConnector& master_connector;
    std::unordered_set<int> available_threads;
    std::unordered_set<Instance, InstanceHasher> instances;
    std::unordered_map<int, std::unordered_set<int>> instance_keeper;
};

}  // namespace husky
