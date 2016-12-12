#pragma once

#include <memory>

#include "husky/core/context.hpp"
#include "worker/basic.hpp"
#include "worker/task_factory.hpp"
#include "worker/worker.hpp"

#include "husky/core/job_runner.hpp"

namespace husky {

/*
 * Engine manages the process
 */
class Engine {
   public:
    Engine() {
        start();
    }
    ~Engine() {
        // TODO Now cannot finalize global, the reason maybe is becuase master_connector still contain
        // the sockets so we cannot delete zmq_context now
        // Context::finalize_global();
    }

    /*
     * Add a new task to the buffer
     */
    void add_task(std::unique_ptr<Task>&& task, const FuncT& func) { worker->add_task(std::move(task), func); }

    /*
     * Submit the buffered tasks to master
     *
     * It's a blocking method, return when all the buffered tasks are finished
     */
    void submit() {
        worker->send_tasks_to_master();
        worker->main_loop();
    }
    /*
     * Ask the Master to exit
     *
     * It means that no more tasks will submit. Basically the end of the process
     */
    void exit() { worker->send_exit(); }

   private:
    /*
     * Start function to initialize the environment
     */
    void start() {
        std::string bind_addr = "tcp://*:" + Context::get_param("worker_port");
        std::string master_addr = "tcp://" + Context::get_config().get_master_host() + ":" +
                                  std::to_string(Context::get_config().get_master_port());
        std::string host_name = Context::get_param("hostname");

        // worker info
        WorkerInfo worker_info = Context::get_worker_info();

        // master connector
        MasterConnector master_connector(*Context::get_zmq_context(), bind_addr, master_addr, host_name);

        // Create mailboxes
        Context::create_mailbox_env();

        // create worker
        worker.reset(new Worker(std::move(worker_info), std::move(master_connector)));
    }

    std::unique_ptr<Worker> worker;
};

}  // namespace husky
