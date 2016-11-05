#pragma once

#include <vector>
#include "base/serialization.hpp"
#include "base/debug.hpp"

namespace husky {

using base::BinStream;

class Cluster {
public:
    Cluster() = default;
    Cluster(const std::vector<int>& threads_)
        :threads(threads_)
    {}

    void set(const std::vector<int>& threads_) {
        threads = threads_;
    }

    void add(int tid) {
        threads.push_back(tid);
    }

    void show_cluster() const {
        std::cout << "[Cluster]: ";
        for (auto th : threads) {
            std::cout << th << " ";
        }
        std::cout << std::endl;
    }

    const auto& get_threads() const {
        return threads;
    }

    // serialization functions
    friend BinStream& operator<<(BinStream& stream, const Cluster& c) {
        stream << c.threads;
        return stream;
    }
    friend BinStream& operator>>(BinStream& stream, Cluster& c) {
        stream >> c.threads;
        return stream;
    }
private:
    std::vector<int> threads;  // global thread ids
};

}
