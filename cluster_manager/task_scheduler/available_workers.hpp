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
    std::vector<std::pair<int, int>> get_workers_per_process(int num_thread_per_worker, int num_processes);

    /*
     * Find required_num_workers in the available workers
     */
    std::vector<std::pair<int, int>> get_workers(int required_num_workers);

    /*
     * Find required_num_workers in one process
     */
    std::vector<std::pair<int, int>> get_local_workers(int required_num_workers);

    /*
     * Gurantee to find required_num_workers in a process which is not frequently visited
     */
    std::vector<std::pair<int, int>> get_traverse_workers(int task_id, int required_num_workers, int num_processes);

    /*
     * Find required_num_workers in an exact process
     */
    std::vector<std::pair<int,int>> get_workers_exact_process(int required_num_workers, int exact_process, int num_processes);

    int get_num_available_workers() {
        return workers_.size();
    }

    int get_num_available_local_workers(int pid) {
        return  pid_tids_[pid].size();
    }

    int get_max_local_workers() {
        int max_num = 0;
        for (auto& kv : pid_tids_)
            max_num = std::max(max_num, static_cast<int>(kv.second.size()));
        return max_num;
    }
        
    void print_available_workers();

   private:
    std::unordered_set<std::pair<int, int>, PairHash> workers_;  // <tid, pid>
    std::unordered_map<int, std::unordered_set<int>> pid_tids_;
};

}  // namespace husky
