#include "core/master.hpp"

using namespace husky;

int main() {
    std::string remote_addr = "proj10";
    std::string remote_port = "12345";

    // worker info
    WorkerInfo worker_info;
    worker_info.add_proc(0, remote_addr);
    worker_info.add_worker(0,0,0);
    worker_info.add_worker(0,1,1);
    worker_info.set_num_processes(1);
    worker_info.set_num_workers(2);
    worker_info.set_proc_id(-1);

    // workers pool
    WorkersPool workers_pool(2);

    // master connection
    zmq::context_t context;
    MasterConnection master_connection(context);
    auto connect_str = "tcp://"+remote_addr+":"+remote_port;
    master_connection.add_proc(0, connect_str);


    Master master(std::move(worker_info),
            std::move(workers_pool),
            std::move(master_connection));
    master.test_connection();
}
