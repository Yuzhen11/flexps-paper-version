#pragma once
#include <memory>

#include "worker/worker.hpp"
#include "core/context.hpp"

namespace husky {

class Engine {
public:
    Engine() {
        start();
    }
    ~Engine() {
        recver.reset();
        el.reset();
        for (int i = 0; i < Context::get_worker_info()->get_num_local_workers(); i++)
            delete mailboxes[i];
        // TODO Now cannot finalize global, the reason maybe is becuase master_connector still contain
        // the sockets so we cannot delete zmq_context now
        // Context::finalize_global();
    }

    template<typename TaskType>
    void add_task(const TaskType& task, const std::function<void(Info)>& func) {
        static_assert(std::is_base_of<Task, TaskType>::value, "TaskType should derived from Task");
        worker->add_task(task, func);
    }

    void run() {
        worker->send_tasks_to_master();
        worker->main_loop();
    }

    template<typename Val>
    int create_kvstore() {
        return worker->create_kvstore<Val>();
    }
private:
    void start() {
        std::string bind_addr = "tcp://*:"+std::to_string(Context::get_config()->get_worker_port());
        std::string master_addr = "tcp://"+Context::get_config()->get_master_host()+":"+std::to_string(Context::get_config()->get_master_port());
        std::string host_name = Context::get_param("hostname");

        // worker info
        WorkerInfo worker_info = *Context::get_worker_info();

        // master connector
        MasterConnector master_connector(Context::get_zmq_context(), bind_addr, master_addr, host_name);

        // Create mailboxes
        el.reset(new MailboxEventLoop(&Context::get_zmq_context()));
        el->set_process_id(worker_info.get_proc_id());
        for (int i = 0; i < worker_info.get_num_processes(); i++)
            el->register_peer_recver(
                i, "tcp://" + worker_info.get_host(i) + ":" + std::to_string(Context::get_config()->get_comm_port()));
        for (int i = 0; i < worker_info.get_num_workers(); i++) {
            if (worker_info.get_proc_id(i) != worker_info.get_proc_id()) {
                el->register_peer_thread(worker_info.get_proc_id(i), i);
            } else {
                auto* mailbox = new LocalMailbox(&Context::get_zmq_context());
                mailbox->set_thread_id(i);
                el->register_mailbox(*mailbox);
                mailboxes.push_back(mailbox);
            }
        }
        recver.reset(new CentralRecver(&Context::get_zmq_context(), Context::get_recver_bind_addr()));
        Context::set_mailboxes(mailboxes);

        // Create mailbox for LocalKVStore
        for (int i = 0; i < worker_info.get_num_processes(); ++ i) {
            int tid = worker_info.get_num_workers() + i;
            if (i != worker_info.get_proc_id()) {
                el->register_peer_thread(worker_info.get_proc_id(tid), tid);
            } else {
                auto* mailbox = new LocalMailbox(&Context::get_zmq_context());
                mailbox->set_thread_id(tid);
                el->register_mailbox(*mailbox);
                Context::set_kv_mailbox(mailbox);
            }
        }

        
        // create worker
        worker.reset(new Worker(std::move(worker_info),
                std::move(master_connector)));
    }

    std::unique_ptr<Worker> worker;
    std::unique_ptr<CentralRecver> recver;
    std::unique_ptr<MailboxEventLoop> el;
    std::vector<LocalMailbox*> mailboxes;
};

}  // namespace husky
