#include <thread>

#include "husky/core/context.hpp"
#include "husky/core/job_runner.hpp"
#include "husky/master/master.hpp"
#include "cluster_manager/cluster_manager_context.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", "serve", "hdfs_namenode", "hdfs_namenode_port"});
    if (!rt)
        return 1;

    // thread for husky::Master
    std::thread master_thread([] {
        auto& master = husky::Master::get_instance();
        master.setup();
        master.serve();
    });

    // thread for ClusterManager
    std::thread cluster_manager_thread([] {
        auto& cluster_manager = husky::ClusterManagerContext::Get();
        cluster_manager.serve();
    });

    // wait for Master and ClusterManager
    master_thread.join();
    cluster_manager_thread.join();

}
