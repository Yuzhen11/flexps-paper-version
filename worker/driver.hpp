#pragma once
#include <memory>

#include "worker/worker.hpp"
#include "core/context.hpp"
#include "kvstore/kvworker.hpp"
#include "kvstore/kvstore_manager.hpp"

namespace husky {

class Engine {
public:
    Engine() {
        start();
    }
    ~Engine() {
        // delete the kvworkers
        for (auto* p : kvworkers) {
            delete p;
        }
        for (auto* p : Context::get_kvworker_mailboxes()) {
            delete p;
        }
        // delete the kvstore_manager 
        kvstore_manager.reset();
        delete Context::get_kvserver_mailbox();

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
        int kv_id = kvstore_manager->create_kvstore<Val>();  // TODO: Ungly
        auto& kvworkers = Context::get_kvworkers();
        for (auto* kvworker : kvworkers) {
            kvworker->AddProcessFunc<Val>(kv_id);
        }
        return kv_id;
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

        int num_workers = worker_info.get_num_workers();
        int num_processes = worker_info.get_num_processes();
        // Create mailboxes
        // mailboxes [0 - num_workers) are for threads communications
        el.reset(new MailboxEventLoop(&Context::get_zmq_context()));
        el->set_process_id(worker_info.get_proc_id());
        for (int i = 0; i < num_processes; i++)
            el->register_peer_recver(
                i, "tcp://" + worker_info.get_host(i) + ":" + std::to_string(Context::get_config()->get_comm_port()));
        for (int i = 0; i < num_workers; i++) {
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

        // The following mailboxes [num_workers - 2*num_workers) are for kvworkers
        std::vector<LocalMailbox*> kv_worker_mailboxes;
        for (int i = 0; i < num_workers; i++) {
            if (worker_info.get_proc_id(i) != worker_info.get_proc_id()) {
                el->register_peer_thread(worker_info.get_proc_id(i), num_workers+i);  // {proc(i), num_workers+i}
            } else {
                auto* mailbox = new LocalMailbox(&Context::get_zmq_context());
                mailbox->set_thread_id(num_workers+i);
                el->register_mailbox(*mailbox);
                kv_worker_mailboxes.push_back(mailbox);
            }
        }
        Context::set_kvworker_mailboxes(kv_worker_mailboxes);

        // The following mailboxes [2*num_workers - 2*num_workers+num_processes) are for kvservers
        for (int i = 0; i < num_processes; ++ i) {
            int tid = 2*num_workers + i;
            if (i != worker_info.get_proc_id()) {
                el->register_peer_thread(worker_info.get_proc_id(tid), tid);
            } else {
                auto* mailbox = new LocalMailbox(&Context::get_zmq_context());
                mailbox->set_thread_id(tid);
                el->register_mailbox(*mailbox);
                Context::set_kvserver_mailbox(mailbox);
            }
        }

        // TODO create kvstore_manager
        kvstore_manager.reset(new kvstore::KVStoreManager(*Context::get_kvserver_mailbox(), constants::kv_channel_id));

        // TODO create kvworkers
        std::unordered_map<int, int> cluster2global;
        for (int i = 0; i < num_processes; ++ i) {
            cluster2global.insert({i, i+2*num_workers});
        }
        for (int i = 0; i < num_workers; ++ i) {
            cluster2global.insert({i+num_processes, i+num_workers});
        }
        int k = 0;
        for (int i = 0; i < num_workers; ++ i) {
            if (worker_info.get_proc_id(i) == worker_info.get_proc_id()) {
                kvstore::PSInfo info;
                info.channel_id = constants::kv_channel_id;
                info.global_id = num_workers + i; 
                info.num_global_threads = num_workers + num_processes;  // workers + servers
                info.num_ps_servers = 1;  // TODO: local_kvstore only need one server
                info.cluster_id_to_global_id = cluster2global;
                kvworkers.push_back(new kvstore::KVWorker(info, *Context::get_kvworker_mailbox(k)));
                k += 1;
            }
        }
        // TODO need to store to Context
        Context::set_kvworkers(kvworkers);

        // create worker
        worker.reset(new Worker(std::move(worker_info),
                std::move(master_connector)));
    }

    std::vector<kvstore::KVWorker*> kvworkers;
    std::unique_ptr<kvstore::KVStoreManager> kvstore_manager;
    std::vector<LocalMailbox*> mailboxes;
    std::unique_ptr<Worker> worker;
    std::unique_ptr<CentralRecver> recver;
    std::unique_ptr<MailboxEventLoop> el;
};

}  // namespace husky
