#include "worker/instance_runner.hpp"

namespace husky {

/*
 * Method to extract local instance
 */
std::vector<std::pair<int, int>> InstanceRunner::extract_local_instance(const std::shared_ptr<Instance>& instance) const {
    auto local_threads = instance->get_threads(worker_info_.get_process_id());
    for (auto& th : local_threads) {
        th.first = worker_info_.global_to_local_id(th.first);
    }
    return local_threads;
}

/*
 * Factory method to generate Info for each running Unit
 */
Info InstanceRunner::info_factory(const std::shared_ptr<Instance>& instance, std::pair<int, int> tid_cid, bool is_leader) {
    Info info = utility::instance_to_info(*instance, worker_info_, tid_cid, is_leader);

    // if TaskType is GenericMLTaskType, set the mlworker according to the instance task type assigned by
    // cluster_manager
    if (info.get_task()->get_type() == Task::Type::MLTaskType) {
        auto& hint = instance->get_task()->get_hint();
        
        try {
            if (hint.at(husky::constants::kType) == husky::constants::kPS) {
                if (hint.find(husky::constants::kWorkerType) != hint.end()
                    && hint.at(husky::constants::kWorkerType) == husky::constants::kSSPWorker) {
                    info.set_mlworker(new ml::mlworker::SSPWorker(info));
                } else {
                    info.set_mlworker(new ml::mlworker::PSWorker(info));
                }
            } else if (hint.at(husky::constants::kType) == husky::constants::kSingle) {
                info.set_mlworker(new ml::mlworker::SingleWorker(info));
            } else if (hint.at(husky::constants::kType) == husky::constants::kHogwild) {
                info.set_mlworker(new ml::mlworker::HogwildWorker(info,
                    cluster_manager_connector_.get_context()));
            } else if (hint.at(husky::constants::kType) == husky::constants::kSPMT) {
                info.set_mlworker(new ml::mlworker::SPMTWorker(info,
                    cluster_manager_connector_.get_context()));
            } else {
                throw;
            }
        } catch(...) {
            utility::print_hint(hint);
            throw base::HuskyException("instance_runner.cpp: Unknown hint");
        }
    }

    return info;
}

/*
 * postprocess function
 */
void InstanceRunner::postprocess(const std::shared_ptr<Instance>& instance, const Info& info) {
    if (info.get_task()->get_type() == Task::Type::MLTaskType) {
        std::string hint = instance->get_task()->get_hint().at(husky::constants::kType);
        if (hint == husky::constants::kSingle || hint == husky::constants::kHogwild || hint == husky::constants::kSPMT) {  // some types need to do dump
            info.get_mlworker()->Dump();
        }
    }
}

/*
 * Run the instances
 */
void InstanceRunner::run_instance(std::shared_ptr<Instance> instance) {
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
    bool is_leader = true;  // the first thread in each process is the leader
    for (auto tid_cid : local_threads) {
        // worker threads must not be joinable (must be free)
        assert(units_[tid_cid.first].joinable() == false);
        units_[tid_cid.first] = boost::thread([this, instance, tid_cid, is_leader] {
            // set the info
            Info info = info_factory(instance, tid_cid, is_leader);

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
        });
        is_leader = false;  // reset is_leader
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
void InstanceRunner::finish_thread(int instance_id, int tid) {
    instance_keeper_[instance_id].erase(tid);
    // husky::LOG_I << "[InstanceRunner]: instance_id: " + std::to_string(instance_id) + " tid: "+
    // std::to_string(tid) + " finished");
    units_[tid].join();
}

bool InstanceRunner::is_instance_done(int instance_id) { 
    return instance_keeper_[instance_id].empty(); 
}

void InstanceRunner::remove_instance(int instance_id) {
    assert(instance_keeper_[instance_id].empty());
    instances_.erase(instance_id);
    instance_keeper_.erase(instance_id);
}

}  // namespace husky
