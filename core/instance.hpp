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
    explicit Instance(int id)
        :id_(id)
    {}
    Instance(int id, int current_epoch)
        :id_(id),
        current_epoch_(current_epoch)
    {}
    Instance(int id, int current_epoch, Task::Type type)
        :id_(id),
        current_epoch_(current_epoch),
        type_(type)
    {}

    void show_instance() const {
        int num_threads = 0;
        for (auto& kv : cluster_)
            num_threads += kv.second.size();
        base::log_msg("[Instance]: Task id:"+std::to_string(id_) + " Epoch:"+std::to_string(current_epoch_) + " Proc Num:"+std::to_string(cluster_.size())+" Thread Num:"+std::to_string(num_threads));
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
        ss << "Task id:" << id_ <<  " Proc id:" << iter->first << ": { ";
        for (auto tid : iter->second) {
            ss << "<" << tid.first << "," << tid.second  << "> ";
        }
        ss << "}";
        base::log_msg("[Instance]: "+ss.str()+" Added");
    }

    // getter
    inline int get_id() const { return id_; }
    inline int get_epoch() const { return current_epoch_; }
    inline Task::Type get_type() const {return type_; }
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
    void set_type(Task::Type type) {
        type_ = type;
    }
    void add_thread(int proc_id, int tid, int id) {
        cluster_[proc_id].push_back({tid, id});
    }
    void set_cluster(const std::unordered_map<int, std::vector<std::pair<int,int>>>& cluster) {
        cluster_ = cluster;
    }

    // serialization functions
    friend BinStream& operator<<(BinStream& stream, const Instance& instance) {
        stream << instance.id_;
        stream << instance.current_epoch_;
        stream << instance.type_;
        stream << instance.cluster_.size();
        for (auto& kv : instance.cluster_) {
            stream << kv.first << kv.second;
        }
        return stream;
    }
    friend BinStream& operator>>(BinStream& stream, Instance& instance) {
        stream >> instance.id_;
        stream >> instance.current_epoch_;
        stream >> instance.type_;
        size_t size;
        stream >> size;
        instance.cluster_.clear();
        for (size_t i = 0; i < size; ++ i) {
            int k;
            std::vector<std::pair<int,int>> v;
            stream >> k >> v;
            instance.cluster_.insert({k, std::move(v)});
        }
        return stream;
    }
private:
    int id_;
    int current_epoch_;
    Task::Type type_;
    std::unordered_map<int, std::vector<std::pair<int,int>>> cluster_;  //  {proc_id, {<tid, id(counting from 0 in this instance)>, ...}}
};
}
