#pragma once

#include "cluster_manager/cluster_manager.hpp"
#include "husky/core/context.hpp"

namespace husky {

/*
 * ClusterManagerContext is just a wrapper for ClusterManager with additional dependency on Context
 *
 * Provide a much easier APIs for user to use as long as Context is initialized
 */
class ClusterManagerContext {
   public:
    /*
     * Singleton Get method
     */
    static ClusterManagerContext& Get() {
        static ClusterManagerContext cluster_manager_context;
        return cluster_manager_context;
    }

    void serve() { cluster_manager_.serve(); }

   private:
    /*
     * Constuctor, assume that Context is initialized
     *
     * Initialize cluster_manager
     */
    ClusterManagerContext() {
        std::string bind_addr = "tcp://*:" + Context::get_param("cluster_manager_port");
        // assume that all remote worker port are the same
        std::string remote_port = Context::get_param("worker_port");

        // worker info
        WorkerInfo worker_info = Context::get_worker_info();

        // cluster_manager connection
        ClusterManagerConnection cluster_manager_connection(Context::get_zmq_context(), bind_addr);

        // connect to remote processes
        auto& procs = worker_info.get_hostnames();
        for (int i = 0; i < procs.size(); ++i) {
            cluster_manager_connection.add_proc(i, "tcp://" + procs[i] + ":" + remote_port);
        }
        std::string task_scheduler_type = Context::get_param("task_scheduler_type");
        cluster_manager_.setup(std::move(worker_info), std::move(cluster_manager_connection), task_scheduler_type);
    }

   private:
    ClusterManager cluster_manager_;
};

}  // namespace husky
