#pragma once

#include <memory>
#include <unordered_map>

#include "husky/base/log.hpp"
#include "husky/core/worker_info.hpp"

#include "ml/mlworker/mlworker.hpp"

namespace husky {

class Task;
class Info {
   public:
    // getter
    int get_local_id() const { return local_id_; }
    int get_global_id() const { return global_id_; }
    int get_cluster_id() const { return cluster_id_; }

    int get_proc_id() const { return worker_info_.get_process_id(); }
    int get_num_local_workers() const { return worker_info_.get_num_local_workers(); }
    int get_num_workers() const { return worker_info_.get_num_workers(); }
    int get_local_pos() const {
        int pos = -1;
        auto local_tids = get_local_tids();
        for (int i = 0; i < local_tids.size(); ++ i) {
            if (local_tids[i] == global_id_) {
                pos = i;
                break;
            }
        }
        assert(pos != -1);
        return pos;
    }
    const WorkerInfo& get_worker_info() const { return worker_info_; }
    WorkerInfo& get_worker_info() { return worker_info_; }
    std::vector<int> get_local_tids() const { return worker_info_.get_local_tids(); }
    std::vector<int> get_pids() const { return worker_info_.get_pids(); }
    int get_current_epoch() const { return current_epoch_; }
    int get_total_epoch() const;
    bool is_leader() const { return is_leader_; }
    const std::map<std::string, std::string>& get_hint() const;

    Task* const get_task() const { return task_; }
    int const get_task_id() const;
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
    void set_task(Task* task) { task_ = task; }
    void set_cluster_global(const std::unordered_map<int, int>& rhs) { cluster_id_to_global_id_ = rhs; }
    void set_cluster_global(std::unordered_map<int, int>&& rhs) { cluster_id_to_global_id_ = std::move(rhs); }
    void set_current_epoch(int current_epoch) { current_epoch_ = current_epoch; }
    void set_leader(bool is_leader) { is_leader_ = is_leader; }

   private:
    int local_id_;
    int global_id_;
    int cluster_id_;  // The id within this cluster

    int current_epoch_;

    WorkerInfo worker_info_;
    Task* task_;

    std::unordered_map<int, int> cluster_id_to_global_id_;  // {cluster_id, global_id}
    bool is_leader_ = false;
};

}  // namespace husky
