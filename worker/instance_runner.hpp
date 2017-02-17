#pragma once

#include <thread>

#include <boost/algorithm/string.hpp>

#include "husky/base/log.hpp"
#include "husky/base/serialization.hpp"
#include "husky/core/worker_info.hpp"
#include "husky/core/zmq_helpers.hpp"

#include "core/info.hpp"
#include "core/instance.hpp"
#include "core/utility.hpp"
#include "worker/cluster_manager_connector.hpp"
#include "worker/task_store.hpp"
#include "worker/unit.hpp"

#include "ml/ml.hpp"

#include "core/color.hpp"

namespace husky {

/*
 * Instances run on threads, InstanceRunner keep track of the
 * instances and threads
 */
class InstanceRunner {
   public:
    InstanceRunner() = delete;
    InstanceRunner(const WorkerInfo& worker_info, ClusterManagerConnector& cluster_manager_connector, TaskStore& task_store)
        : worker_info_(worker_info),
          cluster_manager_connector_(cluster_manager_connector),
          task_store_(task_store),
          units_(worker_info.get_num_local_workers()) {}

    /*
     * Method to extract local instance
     */
    std::vector<std::pair<int, int>> extract_local_instance(const std::shared_ptr<Instance>& instance) const {
        auto local_threads = instance->get_threads(worker_info_.get_process_id());
        for (auto& th : local_threads) {
            th.first = worker_info_.global_to_local_id(th.first);
        }
        return local_threads;
    }

    /*
     * Factory method to generate Info for each running Unit
     */
    Info info_factory(const std::shared_ptr<Instance>& instance, std::pair<int, int> tid_cid) {
        Info info = utility::instance_to_info(*instance, worker_info_, tid_cid);
        info.set_task(task_store_.get_task(instance->get_id()).get());

        // if TaskType is GenericMLTaskType, set the mlworker according to the instance task type assigned by
        // cluster_manager
        if (info.get_task()->get_type() == Task::Type::MLTaskType) {
            std::string hint = instance->get_task()->get_hint();
            
            std::vector<std::string> instructions;
            boost::split(instructions, hint, boost::is_any_of(":"));

            try {
                std::string& first = instructions.at(0);
                if (first == "PS") {
                    std::string& second = instructions.at(1);
                    if (instructions.size() == 2) {
                        info.set_mlworker(new ml::ps::PSGenericWorker(static_cast<MLTask*>(info.get_task())->get_kvstore(),
                                                                      info.get_local_id()));
                    } else if (instructions.at(2) == "SSPWorker") {  // use special SSPWorker to enable thread-cache, e.g. PS:SSP:SSPWorker:2
                        int staleness = std::stoi(instructions.at(3));
                        info.set_mlworker(new ml::ps::SSPWorker(static_cast<MLTask*>(info.get_task())->get_kvstore(), 
                                                                info.get_local_id(),
                                                                staleness));
                    } else {
                        throw;
                    }
                } else if (first == "hogwild") {
                    info.set_mlworker(new ml::hogwild::HogwildGenericWorker(
                        static_cast<MLTask*>(info.get_task())->get_kvstore(), cluster_manager_connector_.get_context(),
                        info, static_cast<MLTask*>(info.get_task())->get_dimensions()));
                    info.get_mlworker()->Load();
                } else if (first == "single") {
                    info.set_mlworker(new ml::single::SingleGenericWorker(
                        static_cast<MLTask*>(info.get_task())->get_kvstore(), info,
                        static_cast<MLTask*>(info.get_task())->get_dimensions()));
                    info.get_mlworker()->Load();
                } else if (first == "SPMT") {
                    std::string& second = instructions.at(1);
                    info.set_mlworker(new ml::spmt::SPMTGenericWorker(
                        static_cast<MLTask*>(info.get_task())->get_kvstore(), cluster_manager_connector_.get_context(),
                        info, second, static_cast<MLTask*>(info.get_task())->get_dimensions()));
                    info.get_mlworker()->Load();
                } else {
                    throw;
                }
            } catch(...) {
                throw base::HuskyException("Unknown hint: " + hint);
            }
            husky::LOG_I << CLAY("[run_instance] set to " + hint);
        }

        return info;
    }

    /*
     * postprocess function
     */
    void postprocess(const std::shared_ptr<Instance>& instance, const Info& info) {
        if (info.get_task()->get_type() == Task::Type::MLTaskType) {
            std::string hint = static_cast<const MLTask*>(instance->get_task())->get_hint();
            if (hint == "single" || hint == "hogwild") {  // some types need to do dump
                info.get_mlworker()->Dump();
            }
        }
    }

    /*
     * Run the instances
     */
    void run_instance(std::shared_ptr<Instance> instance) {
        assert(instances_.find(instance->get_id()) == instances_.end());
        // retrieve local threads
        auto local_threads = extract_local_instance(instance);
        if (local_threads.empty()) {
            return;
        }
        instances_.insert({instance->get_id(), instance});  // store the instance

        husky::LOG_I << GREEN("[InstanceRunner] Instance id " + std::to_string(instance->get_id()) + " " +
                              std::to_string(local_threads.size()) + "/" + std::to_string(instance->get_num_threads()) +
                              " run on process " + std::to_string(worker_info_.get_process_id()));
        for (auto tid_cid : local_threads) {
            // worker threads
            units_[tid_cid.first] = std::move(Unit([this, instance, tid_cid] {
                // set the info
                Info info = info_factory(instance, tid_cid);

                // if (info.get_cluster_id() == 0)
                //     husky::LOG_I << "[Running Task] current_epoch: "+std::to_string(info.get_current_epoch()) + "
                //     starts!";

                // run the UDF!!!
                task_store_.get_func(instance->get_id())(info);

                // postprocess
                postprocess(instance, info);
                // reset the mlworker
                info.get_mlworker().reset();

                // if (info.get_cluster_id() == 0)
                //     husky::LOG_I << "[Running Task] current_epoch: "+std::to_string(info.get_current_epoch()) + "
                //     finishes!";

                // tell worker when I finish
                zmq::socket_t socket = cluster_manager_connector_.get_socket_to_recv();
                zmq_sendmore_int32(&socket, constants::kThreadFinished);
                zmq_sendmore_int32(&socket, instance->get_id());
                zmq_send_int32(&socket, tid_cid.first);
            }));
        }
        std::unordered_set<int> local_threads_set;
        for (auto tid_cid : local_threads)
            local_threads_set.insert(tid_cid.first);
        instance_keeper_.insert({instance->get_id(), std::move(local_threads_set)});
        // husky::LOG_I << "[InstanceRunner]: instance " + std::to_string(instance.get_id()) + " added";
    }

    /*
     * Finish a thread and join the unit
     */
    void finish_thread(int instance_id, int tid) {
        instance_keeper_[instance_id].erase(tid);
        // husky::LOG_I << "[InstanceRunner]: instance_id: " + std::to_string(instance_id) + " tid: "+
        // std::to_string(tid) + " finished");
        units_[tid] = std::move(Unit());  // join the unit
    }
    bool is_instance_done(int instance_id) { return instance_keeper_[instance_id].empty(); }
    void remove_instance(int instance_id) {
        assert(instance_keeper_[instance_id].empty());
        instances_.erase(instance_id);
        instance_keeper_.erase(instance_id);
    }

   private:
    const WorkerInfo& worker_info_;
    ClusterManagerConnector& cluster_manager_connector_;
    TaskStore& task_store_;
    std::unordered_map<int, std::shared_ptr<Instance>> instances_;
    std::unordered_map<int, std::unordered_set<int>> instance_keeper_;
    std::vector<Unit> units_;
};

}  // namespace husky
