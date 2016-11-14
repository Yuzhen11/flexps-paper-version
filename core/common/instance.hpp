#pragma once

#include <sstream>

#include "base/serialization.hpp"
#include "base/debug.hpp"
#include "base/log.hpp"

namespace husky {

using base::BinStream;

class Instance {
public:
    Instance() = default;
    explicit Instance(int id_)
        :id(id_)
    {}
    Instance(int id_, int current_epoch_)
        :id(id_),
        current_epoch(current_epoch_)
    {}

    void show_instance() const {
        int num_threads = 0;
        for (auto& kv : cluster)
            num_threads += kv.second.size();
        base::log_msg("[Instance]: Task id:"+std::to_string(id) + " Epoch:"+std::to_string(current_epoch) + " Proc Num:"+std::to_string(cluster.size())+" Thread Num:"+std::to_string(num_threads));
        for (auto& kv : cluster) {
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
        auto iter = cluster.find(proc_id);
        std::stringstream ss;
        ss << "Task id:" << id <<  " Proc id:" << iter->first << ": { ";
        for (auto tid : iter->second) {
            ss << "<" << tid.first << "," << tid.second  << "> ";
        }
        ss << "}";
        base::log_msg("[Instance]: "+ss.str()+" Added");
    }

    inline int get_id() const {
        return id;
    }
    
    inline int get_epoch() const {
        return current_epoch;
    }

    auto& get_cluster() {
        return cluster;
    }
    const auto& get_cluster() const {
        return cluster;
    }

    void add_thread(int proc_id, int tid, int id) {
        cluster[proc_id].push_back({tid, id});
    }

    auto get_threads(int proc_id) const {
        auto it = cluster.find(proc_id);
        return it->second;
    }

    auto get_num_threads() const {
        int total_threads = 0;
        for (auto& kv : cluster) {
            total_threads += kv.second.size();
        }
        return total_threads;
    }

    auto get_num_procs() const {
        return cluster.size();
    }

    void set_cluster(const std::unordered_map<int, std::vector<std::pair<int,int>>>& cluster_) {
        cluster = cluster_;
    }

    bool operator==(const Instance& rhs) const {
        return id == rhs.get_id();
    }

    // serialization functions
    friend BinStream& operator<<(BinStream& stream, const Instance& instance) {
        stream << instance.id;
        stream << instance.current_epoch;
        stream << instance.cluster.size();
        for (auto& kv : instance.cluster) {
            stream << kv.first << kv.second;
        }
        return stream;
    }
    friend BinStream& operator>>(BinStream& stream, Instance& instance) {
        stream >> instance.id;
        stream >> instance.current_epoch;
        size_t size;
        stream >> size;
        instance.cluster.clear();
        for (size_t i = 0; i < size; ++ i) {
            int k;
            std::vector<std::pair<int,int>> v;
            stream >> k >> v;
            instance.cluster.insert({k, std::move(v)});
        }
        return stream;
    }
private:
    int id;
    int current_epoch;
    std::unordered_map<int, std::vector<std::pair<int,int>>> cluster;  //  {proc_id, {<tid, id(counting from 0 in this instance)>, ...}}
};


struct InstanceHasher {
    std::size_t operator()(const Instance& instance) const {
        return std::hash<int>()(instance.get_id());
    }
};

}
