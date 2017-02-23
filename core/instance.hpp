#pragma once

#include <sstream>

#include "core/color.hpp"
#include "core/task.hpp"
#include "husky/base/log.hpp"
#include "husky/base/serialization.hpp"

namespace husky {

using base::BinStream;

class Instance {
   public:
    Instance() = default;
    Instance(const Task& task);

    // set task
    void set_task(const Task& task, const std::string& hint = "");

    // show function for debug
    void show_instance() const;
    void show_instance(int proc_id) const;

    // getter
    inline int get_id() const { return task_->get_id(); }
    inline int get_epoch() const { return task_->get_current_epoch(); }
    inline int get_num_workers() const { return task_->get_num_workers(); }
    inline Task::Type get_type() const { return task_->get_type(); }
    inline Task* get_task() const { return task_.get(); }
    auto& get_cluster() { return cluster_; }
    const auto& get_cluster() const { return cluster_; }
    std::vector<std::pair<int, int>> get_threads(int proc_id) const;
    int get_num_threads() const;
    auto get_num_procs() const { return cluster_.size(); }

    // setter
    void add_thread(int proc_id, int tid, int id) { cluster_[proc_id].push_back({tid, id}); }
    void set_cluster(const std::unordered_map<int, std::vector<std::pair<int, int>>>& cluster) { cluster_ = cluster; }
    void set_num_workers(int num_workers) { task_->set_num_workers(num_workers); }

    BinStream& serialize(BinStream& bin) const;
    BinStream& deserialize(BinStream& bin);

    // Serialization functions
    friend BinStream& operator<<(BinStream& bin, const Instance& instance) { return instance.serialize(bin); }
    friend BinStream& operator>>(BinStream& bin, Instance& instance) { return instance.deserialize(bin); }

   private:
    std::unique_ptr<Task> task_;
    std::unordered_map<int, std::vector<std::pair<int, int>>>
        cluster_;  //  {proc_id, {<tid, id(counting from 0 in this instance)>, ...}}
};

}  // namespace husky
