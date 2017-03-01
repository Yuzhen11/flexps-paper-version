#pragma once

#include "husky/base/exception.hpp"
#include "husky/base/log.hpp"
#include "husky/core/worker_info.hpp"

#include "cluster_manager/cluster_manager_connection.hpp"
#include "cluster_manager/task_scheduler/task_scheduler.hpp"
#include "cluster_manager/scheduler_trigger.hpp"
#include "core/instance.hpp"

namespace husky {

/*
 * ClusterManager implements the ClusterManager logic
 *
 * Basically it runs an event-loop to receive signal from Workers and react accordingly.
 *
 * The event may be kClusterManagerInit, kClusterManagerInstanceFinished, kClusterManagerExit.
 */
class ClusterManager {
   public:
    ClusterManager() = default;

    void setup(WorkerInfo&& worker_info, ClusterManagerConnection&& cluster_manager_connection, const std::string& hint = "");

    void setup_task_scheduler(const std::string& hint);

    /*
     * The main loop for cluster_manager logic
     */
    void serve();

   private:
    /*
     * Recv tasks from worker and pass the tasks to TaskScheduler
     */
    void recv_tasks_from_worker();

    /*
     * Extract instances and assign them to Workers
     * If no more instances, send exit signal
     */
    void extract_instaces();

    /*
     * Send instances to Workers
     */
    void send_instances(const std::vector<std::shared_ptr<Instance>>& instances);
    /*
     * Possibly enable the ModelTransferManager
     */
    void send_last_instance(const std::shared_ptr<Instance>& instance);

    void set_last_instance(const std::shared_ptr<Instance>& instance);

    /*
     * Send exit signal to Workers
     */
    void send_exit_signal();

   private:
    // store the worker info
    WorkerInfo worker_info_;

    // connect to workers
    std::unique_ptr<ClusterManagerConnection> cluster_manager_connection_;

    // task scheduler
    std::unique_ptr<TaskScheduler> task_scheduler_;

    // decide when to trigger task scheduler
    std::unique_ptr<SchedulerTrigger> scheduler_trigger_; 

    bool is_setup = false;
};

}  // namespace husky
