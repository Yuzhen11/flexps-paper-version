#include <thread>

#include "husky/core/context.hpp"
#include "husky/core/job_runner.hpp"
#include "husky/master/master.hpp"
#include "cluster_manager/cluster_manager.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", "serve", "hdfs_namenode", "hdfs_namenode_port"});
    if (!rt)
        return 1;

    std::string bind_addr = "tcp://*:" + Context::get_param("cluster_manager_port");
    // assume that all remote worker port are the same
    std::string remote_port = Context::get_param("worker_port");

    // worker info
    WorkerInfo worker_info = Context::get_worker_info();

    // cluster_manager connection
    ClusterManagerConnection cluster_manager_connection(*Context::get_zmq_context(), bind_addr);

    auto& procs = worker_info.get_hostnames();
    for (int i = 0; i < procs.size(); ++i) {
        cluster_manager_connection.add_proc(i, "tcp://" + procs[i] + ":" + remote_port);
    }

    // thread for husky::Master
    std::thread master_thread([]() {
        auto& master = husky::Master::get_instance();
        master.setup();
        master.serve();
    });
    ClusterManager cluster_manager(std::move(worker_info), std::move(cluster_manager_connection));
    cluster_manager.cluster_manager_loop();
    master_thread.join();
}
