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

    Instance(const Task& task) { set_task(task); }

    void set_task(const Task& task, const std::string& hint = "") {
        switch (task.get_type()) {
        case Task::Type::BasicTaskType: {  // Basic Task
            task_.reset(new Task(task));
            break;
        }
        case Task::Type::HuskyTaskType: {  // Husky Task
            task_.reset(new HuskyTask(static_cast<const HuskyTask&>(task)));
            break;
        }
        case Task::Type::TwoPhasesTaskType: {  // TwoPhasesTask
            task_.reset(new TwoPhasesTask(static_cast<const TwoPhasesTask&>(task)));
            break;
        }
        case Task::Type::FixedWorkersTaskType: {  // TwoPhasesTask
            task_.reset(new FixedWorkersTask(static_cast<const FixedWorkersTask&>(task)));
            break;
        }
        case Task::Type::MLTaskType: {  // ML Task
            task_.reset(new MLTask(static_cast<const MLTask&>(task)));
            break;
        }
        default:
            throw base::HuskyException("Constructing instance error");
        }
    }

    void show_instance() const {
        task_->show();
        int num_threads = 0;
        for (auto& kv : cluster_)
            num_threads += kv.second.size();
        husky::LOG_I << GREEN("[Instance]: Task id:" + std::to_string(task_->get_id()) + " Epoch:" +
                              std::to_string(task_->get_current_epoch()) + " Proc Num:" +
                              std::to_string(cluster_.size()) + " Thread Num:" + std::to_string(num_threads));
        for (auto& kv : cluster_) {
            std::stringstream ss;
            ss << "Proc id: " << kv.first << ": { ";
            for (auto tid : kv.second) {
                ss << "<" << tid.first << "," << tid.second << "> ";
            }
            ss << "}";
            husky::LOG_I << GREEN("[Instance]: " + ss.str());
        }
    }

    void show_instance(int proc_id) const {
        auto iter = cluster_.find(proc_id);
        if (iter == cluster_.end()) {
            husky::LOG_I << GREEN("No instance added in Proc id: " + std::to_string(proc_id));
            return;
        }
        std::stringstream ss;
        task_->show();
        ss << "Task id:" << task_->get_id() << " Proc id:" << iter->first << ": { ";
        for (auto tid : iter->second) {
            ss << "<" << tid.first << "," << tid.second << "> ";
        }
        ss << "}";
        husky::LOG_I << GREEN("[Instance]: " + ss.str() + " Added");
    }

    // getter
    inline int get_id() const { return task_->get_id(); }
    inline int get_epoch() const { return task_->get_current_epoch(); }
    inline int get_num_workers() const { return task_->get_num_workers(); }
    inline Task::Type get_type() const { return task_->get_type(); }
    inline Task* get_task() const { return task_.get(); }
    auto& get_cluster() { return cluster_; }
    const auto& get_cluster() const { return cluster_; }
    std::vector<std::pair<int, int>> get_threads(int proc_id) const {
        auto it = cluster_.find(proc_id);
        if (it == cluster_.end())
            return {};
        else
            return it->second;
    }
    int get_num_threads() const {
        int total_threads = 0;
        for (auto& kv : cluster_) {
            total_threads += kv.second.size();
        }
        return total_threads;
    }
    auto get_num_procs() const { return cluster_.size(); }

    // setter
    void add_thread(int proc_id, int tid, int id) { cluster_[proc_id].push_back({tid, id}); }
    void set_cluster(const std::unordered_map<int, std::vector<std::pair<int, int>>>& cluster) { cluster_ = cluster; }
    void set_num_workers(int num_workers) { task_->set_num_workers(num_workers); }

    virtual BinStream& serialize(BinStream& bin) const {
        bin << task_->get_type();
        task_->serialize(bin);
        bin << cluster_.size();
        for (auto& kv : cluster_) {
            bin << kv.first << kv.second;
        }
        return bin;
    }
    virtual BinStream& deserialize(BinStream& bin) {
        task_ = std::move(task::deserialize(bin));
        size_t size;
        bin >> size;
        cluster_.clear();
        for (size_t i = 0; i < size; ++i) {
            int k;
            std::vector<std::pair<int, int>> v;
            bin >> k >> v;
            cluster_.insert({k, std::move(v)});
        }
        return bin;
    }
    // Serialization functions
    friend BinStream& operator<<(BinStream& bin, const Instance& instance) { return instance.serialize(bin); }
    friend BinStream& operator>>(BinStream& bin, Instance& instance) { return instance.deserialize(bin); }

   private:
    std::unique_ptr<Task> task_;
    std::unordered_map<int, std::vector<std::pair<int, int>>>
        cluster_;  //  {proc_id, {<tid, id(counting from 0 in this instance)>, ...}}
};
}
