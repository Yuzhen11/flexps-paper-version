#pragma once

#include <unordered_map>
#include "base/log.hpp"
#include "base/serialization.hpp"
#include "core/common/task.hpp"
#include "core/common/instance.hpp"
#include "core/master/task_scheduler.hpp"

namespace husky {

// Use to manage tasks in Master
class ClusterManager {
public:
    ClusterManager() = default;
    explicit ClusterManager(WorkerInfo& worker_info_, MasterConnection& master_connection_)
        : master_connection(master_connection_),
        task_scheduler(new SequentialTaskScheduler(worker_info_)) {
    }

    void init_tasks(const std::vector<Task>& tasks) {
        task_scheduler->init_tasks(tasks);
        for (auto& task : tasks) {
            base::log_msg("[ClusterManager]: Task: "+std::to_string(task.get_task_id())+" added");
        }
    }
    void finish_local_instance(base::BinStream& bin) {
        int instance_id, proc_id;
        bin >> instance_id >> proc_id;
        task_scheduler->finish_local_instance(instance_id, proc_id);
        base::log_msg("[ClusterManager]: instance_id: "+std::to_string(instance_id)+" proc_id: "+std::to_string(proc_id) + " done");
    }

    // try to assign next tasks
    void assign_next_tasks() {
        base::log_msg("[ClusterManager]: Assigning next tasks");
        auto instances = task_scheduler->extract_instances();
        auto& sockets = master_connection.get_send_sockets();
        for (auto& instance : instances) {
            instance.show_instance();
            base::BinStream bin;
            bin << instance;
            auto& cluster = instance.get_cluster();
            for (auto& kv : cluster) {
                auto it = sockets.find(kv.first);
                zmq_sendmore_int32(&it->second, constants::TASK_TYPE);
                zmq_send_binstream(&it->second, bin);
            }
        }
    }

    bool is_finished() {
        return task_scheduler->is_finished();
    }
private:
    MasterConnection& master_connection;

    // task scheduler
    std::unique_ptr<TaskScheduler> task_scheduler;

};
}  // namespace husky
