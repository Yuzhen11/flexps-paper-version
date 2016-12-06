#include "worker/worker.hpp"
#include "core/context.hpp"
#include "worker/task_factory.hpp"

using namespace husky;

int main(int argc, char** argv) {
    Context::init_global();
    bool rt = Context::get_config()->init_with_args(argc, argv, {});
    if (!rt) return 1;

    std::string bind_addr = "tcp://*:"+std::to_string(Context::get_config()->get_worker_port());
    std::string master_addr = "tcp://"+Context::get_config()->get_master_host()+":"+std::to_string(Context::get_config()->get_master_port());
    std::string host_name = Context::get_param("hostname");

    // worker info
    WorkerInfo worker_info = *Context::get_worker_info();

    // master connector
    MasterConnector master_connector(Context::get_zmq_context(), bind_addr, master_addr, host_name);

    // Create mailbox
    auto* el = new MailboxEventLoop(&Context::get_zmq_context());
    el->set_process_id(worker_info.get_proc_id());
    std::vector<LocalMailbox*> mailboxes;
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
    auto* recver = new CentralRecver(&Context::get_zmq_context(), Context::get_recver_bind_addr());
    Context::set_mailboxes(mailboxes);
    
    // create worker
    husky::Worker worker(std::move(worker_info),
            std::move(master_connector));

    // add tasks
    auto task1 = TaskFactory::Get().create_task(Task::Type::BasicTaskType, 1, 2); // id: 0, total_epoch: 1, num_workers: 2
    worker.add_task(std::move(task1), [](const Info& info){
        base::log_msg("local_id:"+std::to_string(info.local_id) + " global_id:" + std::to_string(info.global_id)+" cluster_id:" + std::to_string(info.cluster_id));
        std::this_thread::sleep_for(std::chrono::seconds(1));
        base::log_msg("task1 is running");

        // info.show();
        auto* mailbox = Context::get_mailbox(info.local_id);
        if (info.cluster_id == 0) {  // cluster_id: 0
            // send
            std::string str = "Hello World from cluster id 0";
            BinStream bin;
            bin << str;
            mailbox->send(info.get_tid(1), 0, 0, bin);
            mailbox->send_complete(0,0,&info.hash_ring);

            // recv
            while(mailbox->poll(0,0)) {
                BinStream bin = mailbox->recv(0,0);
                std::string recv;
                bin >> recv;
                base::log_msg("cluster_id:" + std::to_string(info.cluster_id)+" recv: "+recv);
            }
        } else if (info.cluster_id == 1) {  // cluster_id: 1
            // send
            std::string str = "Hello World from cluster id 1";
            BinStream bin;
            bin << str;
            mailbox->send(info.get_tid(0), 0, 0, bin);
            mailbox->send_complete(0,0,&info.hash_ring);

            // recv
            while(mailbox->poll(0,0)) {
                BinStream bin = mailbox->recv(0,0);
                std::string recv;
                bin >> recv;
                base::log_msg("cluster_id:" + std::to_string(info.cluster_id)+" recv: "+recv);
            }
        }
    });

    auto task2 = TaskFactory::Get().create_task(Task::Type::BasicTaskType, 2, 1); // id: 1, total_epoch: 2, num_workers: 1
    worker.add_task(std::move(task2), [](const Info& info){
        base::log_msg("local_id:"+std::to_string(info.local_id) + " global_id:" + std::to_string(info.global_id)+" cluster_id:" + std::to_string(info.cluster_id));
        std::this_thread::sleep_for(std::chrono::seconds(1));
        base::log_msg("task2 is running");
    });

    worker.send_tasks_to_master();
    worker.main_loop();

    // clean up
    delete recver;
    delete el;
    for (int i = 0; i < Context::get_worker_info()->get_num_local_workers(); i++)
        delete mailboxes[i];
    // TODO Now cannot finalize global, the reason maybe is becuase master_connector still contain
    // the sockets so we cannot delete zmq_context now
    // Context::finalize_global();
}
