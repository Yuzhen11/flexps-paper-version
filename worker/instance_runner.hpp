#pragma once

#include <thread>

#include "husky/base/log.hpp"
#include "husky/base/serialization.hpp"
#include "husky/core/worker_info.hpp"
#include "husky/core/zmq_helpers.hpp"

#include "core/info.hpp"
#include "core/instance.hpp"
#include "core/utility.hpp"
#include "worker/master_connector.hpp"
#include "worker/task_store.hpp"
#include "worker/unit.hpp"

#include "ml/ml.hpp"

namespace husky {

/*
 * Instances run on threads, InstanceRunner keep track of the
 * instances and threads
 */
class InstanceRunner {
   public:
    InstanceRunner() = delete;
    InstanceRunner(WorkerInfo& worker_info, MasterConnector& master_connector, TaskStore& task_store)
        : worker_info_(worker_info),
          master_connector_(master_connector),
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

        // if TaskType is GenericMLTaskType, set the mlworker according to the instance task type assigned by master
        if (info.get_task()->get_type() == Task::Type::GenericMLTaskType) {
            base::log_msg("type: " + std::to_string(static_cast<int>(instance->get_type())));
            switch (instance->get_type()) {
            case Task::Type::PSTaskType: {
                throw base::HuskyException("GenericMLTaskType error");
                break;
            }
            case Task::Type::SingleTaskType: {
                base::log_msg("[Debug][run_instance] setting to single generic");
                info.set_mlworker(
                    new ml::single::SingleGenericModel(info.get_task()->get_id(), info.get_local_id(),
                                                       static_cast<GenericMLTask*>(info.get_task())->get_dimensions()));
                info.get_mlworker()->Load();
                break;
            }
            case Task::Type::HogwildTaskType: {
                base::log_msg("[Debug][run_instance] setting to hogwild! generic");
                info.set_mlworker(new ml::hogwild::HogwildGenericModel(
                    info.get_task()->get_id(), master_connector_.get_context(), info,
                    static_cast<GenericMLTask*>(info.get_task())->get_dimensions()));
                info.get_mlworker()->Load();
                break;
            }
            default:
                throw base::HuskyException("GenericMLTaskType error");
            }
        }
        return info;
    }

    /*
     * postprocess function
     */
    void postprocess(const std::shared_ptr<Instance>& instance, const Info& info) {
        if (info.get_task()->get_type() == Task::Type::GenericMLTaskType) {
            switch (instance->get_type()) {
            case Task::Type::PSTaskType: {
                throw base::HuskyException("GenericMLTaskType error");
                break;
            }
            case Task::Type::SingleTaskType: {
                info.get_mlworker()->Dump();
                base::log_msg("[Debug][run_instance] Single generic done");
                break;
            }
            case Task::Type::HogwildTaskType: {
                info.get_mlworker()->Dump();
                base::log_msg("[Debug][run_instance] Hogwild generic done");
                break;
            }
            default:
                throw base::HuskyException("GenericMLTaskType error");
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
        instances_.insert({instance->get_id(), instance});  // store the instance

        // reset the worker for GenericMLTask
        for (auto tid_cid : local_threads) {
            // worker threads
            units_[tid_cid.first] = std::move(Unit([this, instance, tid_cid] {
                zmq::socket_t socket = master_connector_.get_socket_to_recv();
                // set the info
                Info info = info_factory(instance, tid_cid);
                // run the UDF!!!
                task_store_.get_func(instance->get_id())(info);
                postprocess(instance, info);
                info.get_mlworker().reset();
                // tell worker when I finished
                zmq_sendmore_int32(&socket, constants::kThreadFinished);
                zmq_sendmore_int32(&socket, instance->get_id());
                zmq_send_int32(&socket, tid_cid.first);
            }));
        }
        std::unordered_set<int> local_threads_set;
        for (auto tid_cid : local_threads)
            local_threads_set.insert(tid_cid.first);
        instance_keeper_.insert({instance->get_id(), std::move(local_threads_set)});
        // base::log_msg("[InstanceRunner]: instance " + std::to_string(instance.get_id()) + " added");
    }

    /*
     * Finish a thread and join the unit
     */
    void finish_thread(int instance_id, int tid) {
        instance_keeper_[instance_id].erase(tid);
        // base::log_msg("[InstanceRunner]: instance_id: " + std::to_string(instance_id) + " tid: " +
        // std::to_string(tid) + " finished");
        units_[tid] = std::move(Unit());  // join the unit
    }
    bool is_instance_done(int instance_id) { return instance_keeper_[instance_id].empty(); }
    base::BinStream remove_instance(int instance_id) {
        assert(instance_keeper_[instance_id].empty());
        instances_.erase(instance_id);
        instance_keeper_.erase(instance_id);

        // generate the bin to master
        auto proc_id = worker_info_.get_process_id();
        base::BinStream bin;
        bin << instance_id;
        bin << proc_id;
        return bin;
    }

   private:
    WorkerInfo& worker_info_;
    MasterConnector& master_connector_;
    TaskStore& task_store_;
    std::unordered_map<int, std::shared_ptr<Instance>> instances_;
    std::unordered_map<int, std::unordered_set<int>> instance_keeper_;
    std::vector<Unit> units_;
};

}  // namespace husky
