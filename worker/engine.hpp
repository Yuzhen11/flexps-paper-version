#pragma once

#include <memory>

#include "husky/core/context.hpp"
#include "kvstore/kvstore.hpp"
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
        use_kvstore = true;
        start();
    }
    ~Engine() {
        if (use_kvstore) {
            // if use_kvstore stop the kvstore
            kvstore::KVStore::Get().Stop();
        }
        el.reset();
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

    /*
     * Create a new kvstore
     *
     * @return: kvstore id created
     */
    template <typename Val>
    int create_kvstore() {
        assert(use_kvstore);
        return kvstore::KVStore::Get().CreateKVStore<Val>();
    }

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
        // int num_workers = worker_info.get_num_workers();
        // int num_processes = worker_info.get_num_processes();
        // mailboxes [0 - num_workers) are for threads communications
        // el.reset(new MailboxEventLoop(&Context::get_zmq_context()));
        // el->set_process_id(worker_info.get_proc_id());
        // for (int i = 0; i < num_processes; i++)
        //     el->register_peer_recver(
        //         i, "tcp://" + worker_info.get_host(i) + ":" +
        //         std::to_string(Context::get_config()->get_comm_port()));
        // for (int i = 0; i < num_workers; i++) {
        //     if (worker_info.get_proc_id(i) != worker_info.get_proc_id()) {
        //         el->register_peer_thread(worker_info.get_proc_id(i), i);
        //     } else {
        //         auto* mailbox = new LocalMailbox(&Context::get_zmq_context());
        //         mailbox->set_thread_id(i);
        //         el->register_mailbox(*mailbox);
        //         mailboxes.push_back(mailbox);
        //     }
        // }
        // recver.reset(new CentralRecver(&Context::get_zmq_context(), Context::get_recver_bind_addr()));
        // Context::set_mailboxes(mailboxes);

        if (use_kvstore) {
            // if use_kvstore, start the kvstore
            kvstore::KVStore::Get().Start(worker_info, Context::get_mailbox_event_loop(), Context::get_zmq_context());
        }

        // create worker
        worker.reset(new Worker(std::move(worker_info), std::move(master_connector)));
    }

    bool use_kvstore = false;  // whether we need to use kv_store
    std::unique_ptr<Worker> worker;
    std::unique_ptr<MailboxEventLoop> el;
};

}  // namespace husky
