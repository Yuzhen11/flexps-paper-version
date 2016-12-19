#include "cluster_manager/cluster_manager.hpp"

using namespace husky;

int main() {
    std::string remote_addr = "proj10";
    std::string remote_port = "12345";

    // worker info
    WorkerInfo worker_info;
    worker_info.add_worker(0, 0, 0);
    worker_info.add_worker(0, 1, 1);
    worker_info.set_process_id(-1);

    // cluster_manager connection
    std::string bind_addr = "tcp://*:45123";
    zmq::context_t context;
    ClusterManagerConnection cluster_manager_connection(&context, bind_addr);
    auto connect_str = "tcp://" + remote_addr + ":" + remote_port;
    cluster_manager_connection.add_proc(0, connect_str);

    ClusterManager cluster_manager(std::move(worker_info), std::move(cluster_manager_connection));
    // cluster_manager.recv_tasks_from_worker();
    // cluster_manager.test_connection();
    // cluster_manager.assign_initial_tasks();
    cluster_manager.serve();
}
