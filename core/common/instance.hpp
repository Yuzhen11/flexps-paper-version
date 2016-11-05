#pragma once

#include "core/common/cluster.hpp"
#include "base/serialization.hpp"
#include "base/debug.hpp"

namespace husky {

class Instance {
public:
    Instance() = default;
    explicit Instance(int id_, const Cluster& cluster_)
        :id(id_), cluster(cluster_) 
    {}
    explicit Instance(int id_)
        :id(id_)
    {}

    void show_instance() const {
        std::cout << "[Instance]: Instance id: " << id << std::endl;
        cluster.show_cluster();
        std::cout << "[Instance]: Instance end" << std::endl;
    }

    const auto& get_threads() const {
        return cluster.get_threads();
    }
    inline int get_id() const {
        return id;
    }

    auto& get_cluster() {
        return cluster;
    }

    auto get_cluster_size() const {
        return cluster.get_threads().size();
    }

    bool operator==(const Instance& rhs) const {
        return id == rhs.get_id();
    }

    // serialization functions
    friend BinStream& operator<<(BinStream& stream, const Instance& instance) {
        stream << instance.id << instance.cluster;
        return stream;
    }
    friend BinStream& operator>>(BinStream& stream, Instance& instance) {
        stream >> instance.id >> instance.cluster;
        return stream;
    }
private:
    Cluster cluster;
    int id;
};


struct InstanceHasher {
    std::size_t operator()(const Instance& instance) const {
        return std::hash<int>()(instance.get_id());
    }
};

}
