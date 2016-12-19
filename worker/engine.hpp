#pragma once

#include <memory>

#include "husky/core/context.hpp"
#include "worker/basic.hpp"
#include "worker/task_factory.hpp"
#include "worker/worker.hpp"

#include "husky/core/constants.hpp"
#include "husky/core/job_runner.hpp"
#include "husky/core/coordinator.hpp"

namespace husky {

/*
 * Engine manages the process
 */
class Engine {
   public:
    Engine() { 
        start(); 
        // Start Coordinator
        Context::get_coordinator()->serve();
    }
    ~Engine() {
        // TODO Now cannot finalize global, the reason maybe is becuase cluster_manager_connector still contain
        // the sockets so we cannot delete zmq_context now
        // Context::finalize_global();
    }

    /*
     * Add a new task to the buffer
     */
    void add_task(std::unique_ptr<Task>&& task, const FuncT& func) { worker->add_task(std::move(task), func); }

    /*
     * Submit the buffered tasks to cluster_manager
     *
     * It's a blocking method, return when all the buffered tasks are finished
     */
    void submit() {
        worker->send_tasks_to_cluster_manager();
        worker->main_loop();
    }
    /*
     * Ask the ClusterManager to exit
     *
     * It means that no more tasks will submit. Basically the end of the process
     */
    void exit() { 
        worker->send_exit(); 
        // use coordinator to send finish signal
        for (auto tid : Context::get_worker_info().get_local_tids()) {
            base::BinStream finish_signal;
            finish_signal << Context::get_param("hostname") << tid;
            Context::get_coordinator()->notify_master(finish_signal, TYPE_EXIT);
        }
    }

   private:
    /*
     * Start function to initialize the environment
     */
    void start() {
        std::string bind_addr = "tcp://*:" + Context::get_param("worker_port");
        std::string cluster_manager_addr = "tcp://" + Context::get_param("cluster_manager_host") + ":" +
                                  Context::get_param("cluster_manager_port");
        std::string host_name = Context::get_param("hostname");

        // worker info
        WorkerInfo worker_info = Context::get_worker_info();

        // cluster_manager connector
        ClusterManagerConnector cluster_manager_connector(Context::get_zmq_context(), bind_addr, cluster_manager_addr, host_name);

        // Create mailboxes
        Context::create_mailbox_env();

        // create worker
        worker.reset(new Worker(std::move(worker_info), std::move(cluster_manager_connector)));
    }

    std::unique_ptr<Worker> worker;
};

}  // namespace husky
