#pragma once

#include <type_traits>

#include "core/constants.hpp"
#include "core/instance.hpp"
#include "core/worker_info.hpp"
#include "husky/base/exception.hpp"
#include "husky/base/log.hpp"
#include "husky/base/serialization.hpp"
#include "husky/core/zmq_helpers.hpp"
#include "worker/basic.hpp"
#include "worker/cluster_manager_connector.hpp"
#include "worker/instance_runner.hpp"
#include "worker/model_transfer_manager.hpp"
#include "worker/task_store.hpp"

namespace husky {

/*
 * Worker contains the event-loop to receive tasks from ClusterManager,
 * and also has function to communicate with ClusterManager,
 * like send_tasks_to_cluster_manager(), send_exit()
 */
class Worker {
   public:
    Worker() = delete;
    Worker(const WorkerInfo& worker_info_, ModelTransferManager* model_transfer_manager,
           ClusterManagerConnector&& cluster_manager_connector_);

    // User need to add task to taskstore
    template <typename TaskT>
    void add_task(const TaskT& task, const FuncT& func) {
        task_store.add_task(task, func);
    }

    void send_tasks_to_cluster_manager();

    /*
     * send exit signal to cluster_manager, stop the cluster_manager
     * normally it's the last statement in worker
     */
    void send_exit();

    void send_thread_finished(int instance_id, int thread_id);

    void main_loop();

   private:
    ClusterManagerConnector cluster_manager_connector;
    const WorkerInfo& worker_info;
    ModelTransferManager* model_transfer_manager_;
    InstanceRunner instance_runner;
    TaskStore task_store;
};

}  // namespace husky
