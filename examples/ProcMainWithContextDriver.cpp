#include "core/worker/driver.hpp"

using namespace husky;

int main(int argc, char** argv) {
    Context::init_global();
    bool rt = Context::get_config()->init_with_args(argc, argv, {});
    if (!rt) return 1;

    Engine engine;
    engine.create_worker();

    // add tasks
    Task task1(0,1,2);  // id: 0, total_epoch: 1, num_workers: 2
    engine.add_task(task1, [](Info info){
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

    Task task2(1,3,4);  // id: 1, total_epoch: 2, num_workers: 1
    engine.add_task(task2, [](Info info){
        base::log_msg("local_id:"+std::to_string(info.local_id) + " global_id:" + std::to_string(info.global_id)+" cluster_id:" + std::to_string(info.cluster_id));
        std::this_thread::sleep_for(std::chrono::seconds(1));
        base::log_msg("task2 is running");
    });

    engine.run();

    return 0;
}
