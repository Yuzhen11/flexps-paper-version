#pragma once

#include <memory>
#include <unordered_map>

#include "core/task.hpp"
#include "husky/base/log.hpp"
#include "husky/core/worker_info.hpp"

#include "ml/mlworker/mlworker.hpp"

namespace husky {

class Info {
   public:
    // getter
    int get_local_id() const { return local_id_; }
    int get_global_id() const { return global_id_; }
    int get_cluster_id() const { return cluster_id_; }

    int get_proc_id() const { return worker_info_.get_process_id(); }
    int get_num_local_workers() const { return worker_info_.get_num_local_workers(); }
    int get_num_workers() const { return worker_info_.get_num_workers(); }
    const WorkerInfo& get_worker_info() const { return worker_info_; }
    WorkerInfo& get_worker_info() { return worker_info_; }
    std::vector<int> get_local_tids() const { return worker_info_.get_local_tids(); }
    std::vector<int> get_pids() const { return worker_info_.get_pids(); }
    int get_current_epoch() const { return current_epoch_; }
    int get_total_epoch() const { return task_->get_total_epoch(); }

    const std::unique_ptr<ml::mlworker::GenericMLWorker>& get_mlworker() const { return mlworker_; }
    std::unique_ptr<ml::mlworker::GenericMLWorker>& get_mlworker() { return mlworker_; }
    Task* const get_task() const { return task_; }
    inline int const get_task_id() const { return task_->get_id(); }
    const std::unordered_map<int, int>& get_cluster_global() const { return cluster_id_to_global_id_; }

    int get_tid(int cluster_id) const {
        auto p = cluster_id_to_global_id_.find(cluster_id);
        return p->second;
    }
    void show() const {
        for (auto& kv : cluster_id_to_global_id_) {
            husky::LOG_I << "Info: " + std::to_string(kv.first) + " " + std::to_string(kv.second);
        }
    }

    // setter
    void set_local_id(int local_id) { local_id_ = local_id; }
    void set_global_id(int global_id) { global_id_ = global_id; }
    void set_cluster_id(int cluster_id) { cluster_id_ = cluster_id; }
    void set_worker_info(const WorkerInfo& worker_info) { worker_info_ = worker_info; }
    void set_worker_info(WorkerInfo&& worker_info) { worker_info_ = std::move(worker_info); }
    void set_mlworker(ml::mlworker::GenericMLWorker* p) { mlworker_.reset(p); }
    void set_task(Task* task) { task_ = task; }
    void set_cluster_global(const std::unordered_map<int, int>& rhs) { cluster_id_to_global_id_ = rhs; }
    void set_cluster_global(std::unordered_map<int, int>&& rhs) { cluster_id_to_global_id_ = std::move(rhs); }
    void set_current_epoch(int current_epoch) { current_epoch_ = current_epoch; }

   private:
    int local_id_;
    int global_id_;
    int cluster_id_;  // The id within this cluster

    int current_epoch_;

    WorkerInfo worker_info_;
    Task* task_;
    std::unique_ptr<ml::mlworker::GenericMLWorker> mlworker_;

    std::unordered_map<int, int> cluster_id_to_global_id_;  // {cluster_id, global_id}
};

}  // namespace husky
