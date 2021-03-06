#include <thread>
#include <boost/thread/thread.hpp>

#include "cluster_manager/cluster_manager_context.hpp"
#include "husky/core/context.hpp"
#include "husky/core/job_runner.hpp"
#include "husky/master/master.hpp"

// read hdfs_block
#include "io/hdfs_assigner_ml.hpp"
// read hdfs_binary
#include "io/hdfs_binary_assigner_ml.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", "serve",
                                          "hdfs_namenode", "hdfs_namenode_port", "task_scheduler_type", 
                                          "scheduler_trigger_time_out", "scheduler_trigger_num_threads"});
    if (!rt)
        return 1;

    // thread for husky::Master
    std::thread master_thread([] {
        auto& master = husky::Master::get_instance();
        
        // finish some handlers
        HDFSBlockAssignerML hdfs_block_assign;

        // finish some handlers
        HDFSFileAssignerML hdfs_binary_assigner;
        
        master.setup();
        master.serve();
    });

    // thread for ClusterManager
    boost::thread cluster_manager_thread([] {
        auto& cluster_manager = husky::ClusterManagerContext::Get();
        cluster_manager.serve();
    });

    // wait for Master and ClusterManager
    master_thread.join();
    cluster_manager_thread.join();
}
