#pragma once

#include <sstream>

#include "base/serialization.hpp"
#include "base/debug.hpp"
#include "base/log.hpp"
#include "core/task.hpp"

namespace husky {

using base::BinStream;

class Instance {
public:
    Instance() = default;

    Instance(Task& task, Task::Type newtype = Task::Type::DummyType) {
        set_task(task, newtype);
    }

    void set_task(Task& task, Task::Type newtype = Task::Type::DummyType) {
        switch(task.get_type()) {
            case Task::Type::BasicTaskType: {   // Basic Task
                task_.reset(new Task(task));
                break;
            }
            case Task::Type::HuskyTaskType: {  // Husky Task
                task_.reset(new HuskyTask(static_cast<HuskyTask&>(task)));
                break;
            }
            case Task::Type::PSTaskType: {  // PS Task
                task_.reset(new PSTask(static_cast<PSTask&>(task)));
                break;
            }
            case Task::Type::HogwildTaskType: {  // Hogwild Task
                task_.reset(new HogwildTask(static_cast<HogwildTask&>(task)));
                break;
            }
            case Task::Type::SingleTaskType: {  // Single Task
                task_.reset(new SingleTask(static_cast<SingleTask&>(task)));
                break;
            }
            case Task::Type::GenericMLTaskType: {  // GenericML Task
                assert(newtype != Task::Type::DummyType);
                switch(newtype) {
                    case Task::Type::PSTaskType: {
                        task_.reset(new PSTask());
                        break;
                    }
                    case Task::Type::HogwildTaskType: {
                        task_.reset(new HogwildTask());
                        break;
                    }
                    case Task::Type::SingleTaskType: {
                        task_.reset(new SingleTask());
                        break;
                    }
                    default:
                        throw base::HuskyException("Constructing instance error");
                }
                task_->set_id(task.get_id());
                task_->set_total_epoch(task.get_total_epoch());
                task_->set_current_epoch(task.get_current_epoch());
                task_->set_num_workers(task.get_num_workers());
                break;
            }
            default:
                throw base::HuskyException("Constructing instance error");
        }
    }

    void show_instance() const {
        int num_threads = 0;
        for (auto& kv : cluster_)
            num_threads += kv.second.size();
        base::log_msg("[Instance]: Task id:"+std::to_string(task_->get_id()) + " Epoch:"+std::to_string(task_->get_current_epoch()) + " Proc Num:"+std::to_string(cluster_.size())+" Thread Num:"+std::to_string(num_threads));
        for (auto& kv : cluster_) {
            std::stringstream ss;
            ss << "Proc id: " << kv.first << ": { ";
            for (auto tid : kv.second) {
                ss << "<" << tid.first << "," << tid.second  << "> ";
            }
            ss << "}";
            base::log_msg("[Instance]: "+ss.str());
        }
    }

    void show_instance(int proc_id) const {
        auto iter = cluster_.find(proc_id);
        std::stringstream ss;
        ss << "Task id:" << task_->get_id() <<  " Proc id:" << iter->first << ": { ";
        for (auto tid : iter->second) {
            ss << "<" << tid.first << "," << tid.second  << "> ";
        }
        ss << "}";
        base::log_msg("[Instance]: "+ss.str()+" Added");
    }

    // getter
    inline int get_id() const { return task_->get_id(); }
    inline int get_epoch() const { return task_->get_current_epoch(); }
    inline Task::Type get_type() const {return task_->get_type(); }
    auto& get_cluster() { return cluster_; }
    const auto& get_cluster() const { return cluster_; }
    auto get_threads(int proc_id) const {
        auto it = cluster_.find(proc_id);
        return it->second;
    }
    auto get_num_threads() const {
        int total_threads = 0;
        for (auto& kv : cluster_) {
            total_threads += kv.second.size();
        }
        return total_threads;
    }
    auto get_num_procs() const {
        return cluster_.size();
    }

    // setter
    void add_thread(int proc_id, int tid, int id) {
        cluster_[proc_id].push_back({tid, id});
    }
    void set_cluster(const std::unordered_map<int, std::vector<std::pair<int,int>>>& cluster) {
        cluster_ = cluster;
    }

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
        for (size_t i = 0; i < size; ++ i) {
            int k;
            std::vector<std::pair<int,int>> v;
            bin >> k >> v;
            cluster_.insert({k, std::move(v)});
        }
        return bin;
    }
    // Serialization functions
    friend BinStream& operator<<(BinStream& bin, const Instance& instance) {
        return instance.serialize(bin);
    }
    friend BinStream& operator>>(BinStream& bin, Instance& instance) {
        return instance.deserialize(bin);
    }
private:
    std::unique_ptr<Task> task_;
    std::unordered_map<int, std::vector<std::pair<int,int>>> cluster_;  //  {proc_id, {<tid, id(counting from 0 in this instance)>, ...}}
};

}
