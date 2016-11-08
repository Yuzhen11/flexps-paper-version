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
        base::log_msg("[Instance]: Instance id: "+std::to_string(id) + " epoch: "+std::to_string(current_epoch));
        for (auto& kv : cluster) {
            std::stringstream ss;
            ss << "Proc id: " << kv.first << ": { ";
            for (auto tid : kv.second) {
                ss << tid << " ";
            }
            ss << "}";
            base::log_msg("[Instance]: "+ss.str());
        }
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

    void add_thread(int proc_id, int tid) {
        cluster[proc_id].push_back(tid);
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

    void set_cluster(const std::unordered_map<int, std::vector<int>>& cluster_) {
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
            std::vector<int> v;
            stream >> k >> v;
            instance.cluster.insert({k, std::move(v)});
        }
        return stream;
    }
private:
    int id;
    int current_epoch;
    std::unordered_map<int, std::vector<int>> cluster;  //  {proc_id, {tid...}}
};


struct InstanceHasher {
    std::size_t operator()(const Instance& instance) const {
        return std::hash<int>()(instance.get_id());
    }
};

}
