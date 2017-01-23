#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace husky {

struct PairHash {
    template <typename T, typename U>
    std::size_t operator()(const std::pair<T, U>& x) const {
        return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
    }
};

/*
 * AvailableWorkers keep track of the available workers
 */
class AvailableWorkers {
   public:
    void add_worker(int pid, int tid) {
        workers_.insert({tid, pid});
        pid_tids_[pid].insert(tid);
    }

    void remove_worker(int pid, int tid) {
        workers_.erase({tid, pid});
        pid_tids_[pid].erase(tid);
    }

    /*
     * Find the same number of workers in each process
     */
    std::vector<std::pair<int, int>> get_workers_per_process(int num_thread_per_worker, int num_processes) {
        std::map<int, std::vector<int>> pid_tids_map;
        // init map
        for (int i = 0; i < num_processes; i++) {
            pid_tids_map.emplace(i, std::vector<int>());
        }

        for (auto tid_pid : workers_) {
            pid_tids_map[tid_pid.second].push_back(tid_pid.first);
        }

        // Guarantee at least thread_per_worker threads in each worker
        for (auto pid_tid_map : pid_tids_map) {
            if (pid_tid_map.second.size() < num_thread_per_worker) {
                return {};
            }
        }

        // The requirement is satisfied, get these threads
        std::vector<std::pair<int, int>> selected_workers;
        for (auto pid_tid_map : pid_tids_map) {
            for (int j = 0; j < num_thread_per_worker; j++) {
                selected_workers.push_back({pid_tid_map.first, pid_tid_map.second[j]});  // <pid, tid>
            }
        }

        // erase from workers
        for (auto pid_tid : selected_workers) {
            workers_.erase({pid_tid.second, pid_tid.first});
        }
        // erase from pid_tids
        for (auto pid_tid : selected_workers)
            pid_tids_[pid_tid.first].erase(pid_tid.second);

        return selected_workers;
    }
    /*
     * Find required_num_workers in the available workers
     */
    std::vector<std::pair<int, int>> get_workers(int required_num_workers) {
        if (workers_.size() < required_num_workers)
            return {};
        // Since all the workers are stored in unordered_set,
        // I just get the first *required_num_workers* threads
        std::vector<std::pair<int, int>> selected_workers;
        auto it = workers_.begin();
        while (selected_workers.size() < required_num_workers) {
            selected_workers.push_back({it->second, it->first});  // <pid, tid>
            ++it;
        }
        // erase from workers_
        workers_.erase(workers_.begin(), it);
        // erase from pid_tids_
        for (auto pid_tid : selected_workers)
            pid_tids_[pid_tid.first].erase(pid_tid.second);
        return selected_workers;
    }

    /*
     * Find required_num_workers in one process
     */
    std::vector<std::pair<int, int>> get_local_workers(int required_num_workers) {
        // find the first process that contains no less than *required_num_workers* 's threads
        for (auto& kv : pid_tids_) {
            if (kv.second.size() >= required_num_workers) {
                std::vector<std::pair<int, int>> selected_workers;
                auto it = kv.second.begin();
                while (selected_workers.size() < required_num_workers) {
                    selected_workers.push_back({kv.first, *it});
                    ++it;
                }
                // erase from workers
                for (auto pid_tid : selected_workers) {
                    workers_.erase({pid_tid.second, pid_tid.first});
                }
                // erase from pid_tids_
                kv.second.erase(kv.second.begin(), it);
                return selected_workers;
            }
        }
        return {};
    }

    int get_num_available_workers() { return workers_.size(); }

    int get_max_local_workers() {
        int max_num = 0;
        for (auto& kv : pid_tids_)
            max_num = std::max(max_num, static_cast<int>(kv.second.size()));
        return max_num;
    }

   private:
    std::unordered_set<std::pair<int, int>, PairHash> workers_;  // <tid, pid>
    std::unordered_map<int, std::unordered_set<int>> pid_tids_;
};

}  // namespace husky
