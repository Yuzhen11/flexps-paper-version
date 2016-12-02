#include "master/master.hpp"

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

    // master connection
    std::string bind_addr = "tcp://*:45123";
    zmq::context_t context;
    MasterConnection master_connection(context, bind_addr);
    auto connect_str = "tcp://"+remote_addr+":"+remote_port;
    master_connection.add_proc(0, connect_str);


    Master master(std::move(worker_info),
            std::move(master_connection));
    // master.recv_tasks_from_worker();
    // master.test_connection();
    // master.assign_initial_tasks();
    master.master_loop();
}
