#pragma once

#include <vector>
#include <unordered_set>
#include <cassert>

namespace husky {

class WorkersPool {
public:
    WorkersPool() = default;
    WorkersPool(int num_workers_)
    : num_workers(num_workers_) {
        for (int i = 0; i < num_workers; ++ i)
            available_workers.insert(i);
    }

    // allocate workers from workers_pool
    void use_workers(const std::vector<int>& workers) {
        for (auto& w : workers) {
            assert(available_workers.find(w) != available_workers.end());
            available_workers.erase(w);
        }
    }

    // deallocate workers from workers_pool
    void free_workers(const std::vector<int>& workers) {
        for (auto& w : workers) {
            assert(available_workers.find(w) == available_workers.end());
            available_workers.insert(w);
        }
    }

    const std::unordered_set<int>& get_available_workers() {
        return available_workers;
    }

private:
    int num_workers = -1;
    std::unordered_set<int> available_workers;
};

}  // namespace husky
