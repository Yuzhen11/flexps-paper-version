#include "worker/worker.hpp"
#include "core/context.hpp"
#include "worker/task_factory.hpp"
#include "husky/core/job_runner.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port"});
    if (!rt) return 1;

    std::string bind_addr = "tcp://*:"+Context::get_param("worker_port");
    std::string master_addr = "tcp://"+Context::get_config().get_master_host()+":"+std::to_string(Context::get_config().get_master_port());
    std::string host_name = Context::get_param("hostname");

    // worker info
    WorkerInfo worker_info = Context::get_worker_info();

    // master connector
    MasterConnector master_connector(*Context::get_zmq_context(), bind_addr, master_addr, host_name);

    // Create mailbox
    Context::create_mailbox_env();
    
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
            mailbox->send_complete(0,0,info.local_tids, info.global_pids);

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
            mailbox->send_complete(0,0,info.local_tids, info.global_pids);

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
    // delete recver;
    // delete el;
    // for (int i = 0; i < Context::get_worker_info()->get_num_local_workers(); i++)
    //     delete mailboxes[i];
    // TODO Now cannot finalize global, the reason maybe is becuase master_connector still contain
    // the sockets so we cannot delete zmq_context now
    // Context::finalize_global();
}
