#pragma once

#include <thread>

#include "base/debug.hpp"
#include "base/log.hpp"
#include "base/serialization.hpp"

#include "core/instance.hpp"
#include "core/worker_info.hpp"
#include "core/zmq_helpers.hpp"
#include "core/info.hpp"
#include "core/utility.hpp"
#include "worker/master_connector.hpp"
#include "worker/task_store.hpp"
#include "worker/unit.hpp"

namespace husky {

/*
 * Instances run on threads, InstanceRunner keep track of the 
 * instances and threads
 */
class InstanceRunner {
public:
    InstanceRunner() = delete;
    InstanceRunner(WorkerInfo& worker_info, MasterConnector& master_connector, TaskStore& task_store)
        : worker_info_(worker_info),
        master_connector_(master_connector),
        task_store_(task_store),
        units_(worker_info.get_num_local_workers())
    {}

    std::vector<std::pair<int,int>> extract_local_instance(const Instance& instance) const {
        auto local_threads = instance.get_threads(worker_info_.get_proc_id());
        for (auto& th : local_threads) {
            th.first = worker_info_.global_to_local_id(th.first);
        }
        return local_threads;
    }

    void run_instance(const Instance& instance) {
        assert(instances_.find(instance.get_id()) == instances_.end());
        // retrieve local threads
        auto local_threads = extract_local_instance(instance);
        instances_.insert({instance.get_id(), instance});  // store the instance

        for (auto tid_cid : local_threads) {
            // worker threads
            units_[tid_cid.first] = std::move(Unit([this, instance, tid_cid]{
                zmq::socket_t socket = master_connector_.get_socket_to_recv();
                // set the info
                Info info = utility::instance_to_info(instance);
                info.local_id = tid_cid.first;
                info.global_id = worker_info_.local_to_global_id(tid_cid.first);
                info.cluster_id = tid_cid.second;
                info.num_local_threads = instance.get_threads(worker_info_.get_proc_id()).size();
                info.num_global_threads = instance.get_num_threads();
                info.task = task_store_.get_task(instance.get_id());
                // run the UDF!!!
                task_store_.get_func(instance.get_id())(info);
                // tell worker when I finished
                zmq_sendmore_int32(&socket, constants::THREAD_FINISHED);
                zmq_sendmore_int32(&socket, instance.get_id());
                zmq_send_int32(&socket, tid_cid.first);
            }));
        }
        std::unordered_set<int> local_threads_set;
        for (auto tid_cid : local_threads)
            local_threads_set.insert(tid_cid.first);
        instance_keeper_.insert({instance.get_id(), std::move(local_threads_set)});
        // base::log_msg("[InstanceRunner]: instance " + std::to_string(instance.get_id()) + " added");
    }

    void finish_thread(int instance_id, int tid) {
        instance_keeper_[instance_id].erase(tid);
        // base::log_msg("[InstanceRunner]: instance_id: " + std::to_string(instance_id) + " tid: " + std::to_string(tid) + " finished");
        units_[tid] = std::move(Unit());  // join the unit
    }
    bool is_instance_done(int instance_id) {
        return instance_keeper_[instance_id].empty();
    }
    void remove_instance(int instance_id) {
        assert(instance_keeper_[instance_id].empty());
        instances_.erase(instance_id);
        instance_keeper_.erase(instance_id);

        // tell master
        auto proc_id = worker_info_.get_proc_id();
        base::BinStream bin;
        bin << instance_id;
        bin << proc_id;
        auto& socket = master_connector_.get_send_socket();
        zmq_send_binstream(&socket, bin);  // {instance_id, proc_id}
    }

private:
    WorkerInfo& worker_info_;
    MasterConnector& master_connector_;
    TaskStore& task_store_;
    std::unordered_map<int, Instance> instances_;
    std::unordered_map<int, std::unordered_set<int>> instance_keeper_;
    std::vector<Unit> units_;
};

}  // namespace husky
