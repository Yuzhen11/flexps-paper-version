#pragma once

#include <boost/thread/thread.hpp>
#include <thread>

#include "husky/base/log.hpp"
#include "husky/base/serialization.hpp"
#include "husky/core/worker_info.hpp"
#include "husky/core/zmq_helpers.hpp"

#include "core/info.hpp"
#include "core/instance.hpp"
#include "core/utility.hpp"
#include "worker/cluster_manager_connector.hpp"
#include "worker/task_store.hpp"

#include "core/color.hpp"

namespace husky {

/*
 * Instances run on threads, InstanceRunner keep track of the
 * instances and threads
 */
class InstanceRunner {
   public:
    InstanceRunner() = delete;
    InstanceRunner(const WorkerInfo& worker_info, ClusterManagerConnector& cluster_manager_connector,
                   TaskStore& task_store)
        : worker_info_(worker_info),
          cluster_manager_connector_(cluster_manager_connector),
          task_store_(task_store),
          units_(worker_info.get_num_local_workers()) {}

    /*
     * Method to extract local instance
     */
    std::vector<std::pair<int, int>> extract_local_instance(const std::shared_ptr<Instance>& instance) const;
    /*
     * Run the instances
     */
    void run_instance(std::shared_ptr<Instance> instance);
    /*
     * Finish a thread and join the unit
     */
    void finish_thread(int instance_id, int tid);
    bool is_instance_done(int instance_id);
    void remove_instance(int instance_id);

   private:
    void assign_worker(const std::pair<int, int>& tid_cid, std::shared_ptr<Instance> instance, bool is_leader);
    const WorkerInfo& worker_info_;
    ClusterManagerConnector& cluster_manager_connector_;
    TaskStore& task_store_;
    std::unordered_map<int, std::shared_ptr<Instance>> instances_;
    std::unordered_map<int, std::unordered_set<int>> instance_keeper_;
    std::vector<boost::thread> units_;
};

}  // namespace husky
