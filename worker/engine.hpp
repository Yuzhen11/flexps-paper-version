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
    static Engine& Get() {
        static Engine engine;
        return engine;
    }

    /*
     * Add a new task to the buffer
     */
    template<typename TaskT>
    void AddTask(const TaskT& task, const FuncT& func) { worker->add_task(task, func); }

    /*
     * Submit the buffered tasks to cluster_manager
     *
     * It's a blocking method, return when all the buffered tasks are finished
     */
    void Submit() {
        worker->send_tasks_to_cluster_manager();
        worker->main_loop();
    }
    /*
     * Ask the ClusterManager to exit
     *
     * It means that no more tasks will submit. Basically the end of the process
     */
    void Exit() { 
        StopWorker();
        StopCoordinator();
    }

   private:

    // The constructor
    Engine() { 
        StartWorker(); 
        StartCoordinator();
    }

    /*
     * Start function to initialize the environment
     */
    void StartWorker() {
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

    void StartCoordinator() {
        Context::get_coordinator()->serve();
    }

    // Function to stop the worker
    void StopWorker() {
        worker->send_exit(); 
    }

    // Function to stop the coordinator
    void StopCoordinator() {
        for (auto tid : Context::get_worker_info().get_local_tids()) {
            base::BinStream finish_signal;
            finish_signal << Context::get_param("hostname") << tid;
            Context::get_coordinator()->notify_master(finish_signal, TYPE_EXIT);
        }
    }

    std::unique_ptr<Worker> worker;
};

}  // namespace husky
