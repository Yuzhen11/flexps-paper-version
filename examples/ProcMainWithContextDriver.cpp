#include "husky/core/job_runner.hpp"
#include "worker/engine.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();

    // add tasks
    auto task1 =
        TaskFactory::Get().CreateTask<Task>(1, 2);  // id: 0, total_epoch: 1, num_workers: 2
    engine.AddTask(task1, [](const Info& info) {
        LOG_I << "local_id:" + std::to_string(info.get_local_id()) + " global_id:" +
                      std::to_string(info.get_global_id()) + " cluster_id:" + std::to_string(info.get_cluster_id());
        std::this_thread::sleep_for(std::chrono::seconds(1));
        LOG_I << "task1 is running";

        // info.show();
        auto* mailbox = Context::get_mailbox(info.get_local_id());
        if (info.get_cluster_id() == 0) {  // cluster_id: 0
            // send
            std::string str = "Hello World from cluster id 0";
            BinStream bin;
            bin << str;
            mailbox->send(info.get_tid(1), 0, 0, bin);
            mailbox->send_complete(0, 0, info.get_local_tids(), info.get_pids());

            // recv
            while (mailbox->poll(0, 0)) {
                BinStream bin = mailbox->recv(0, 0);
                std::string recv;
                bin >> recv;
                LOG_I << "cluster_id:" + std::to_string(info.get_cluster_id()) + " recv: " + recv;
            }
        } else if (info.get_cluster_id() == 1) {  // cluster_id: 1
            // send
            std::string str = "Hello World from cluster id 1";
            BinStream bin;
            bin << str;
            mailbox->send(info.get_tid(0), 0, 0, bin);
            mailbox->send_complete(0, 0, info.get_local_tids(), info.get_pids());

            // recv
            while (mailbox->poll(0, 0)) {
                BinStream bin = mailbox->recv(0, 0);
                std::string recv;
                bin >> recv;
                LOG_I << "cluster_id:" + std::to_string(info.get_cluster_id()) + " recv: " + recv;
            }
        }
    });

    auto task2 =
        TaskFactory::Get().CreateTask<Task>(2, 1);  // id: 1, total_epoch: 2, num_workers: 1
    engine.AddTask(task2, [](const Info& info) {
        LOG_I << "local_id:" + std::to_string(info.get_local_id()) + " global_id:" +
                      std::to_string(info.get_global_id()) + " cluster_id:" + std::to_string(info.get_cluster_id());
        std::this_thread::sleep_for(std::chrono::seconds(1));
        LOG_I << "task2 is running";
    });

    engine.Submit();
    engine.Exit();

    return 0;
}
