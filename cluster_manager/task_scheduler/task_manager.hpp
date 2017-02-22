#pragma once

#include <list>
#include <unordered_set> 
#include <unordered_map> 

#include "available_workers.hpp"
#include "history_manager.hpp" 
#include "core/color.hpp"

namespace husky {
// considering both the history and priority

class TaskManager {
   public:
    TaskManager() {}
    TaskManager(const TaskManager&) = delete;

    void add_tasks(const std::vector<std::shared_ptr<Task>>& tasks);

    int get_num_tasks();

    std::shared_ptr<Task> get_task_by_id(int task_id);

    int get_task_priority(int task_id);

    int get_task_status(int task_id);

    int get_task_rej_times(int task_id);
    
    std::unordered_set<int> get_tracking_threads(int task_id, int proc_id);

    std::list<int>::const_iterator angry_list_begin() {return angry_list_.begin(); }

    std::list<int>::const_iterator angry_list_end() {return angry_list_.end(); }

    void erase_angry_task(int task_id);

    bool exist_angry_tasks();

    bool is_finished();

    void finish_thread(int task_id, int pid, int global_thread_id);

    std::vector<int> order_by_priority();

    std::vector<int> get_preferred_proc(int task_id);

    void suc_sched(int task_id);

    void fail_sched(int task_id);

    void record_and_track(int task_id, std::vector<std::pair<int, int>>& pid_tid);

   private: 
    bool in_angry_list(int task_id);
    
   private:
    int angry_threshold_ = 5;
    int num_tasks_ = 0;
    std::vector<std::shared_ptr<Task>> tasks_;
    std::vector<std::pair<int, int>> task_priority_; // vector of <task_id, priority> 
    std::vector<int> task_status_; // 0: ready; 1: running; 2: finished; 3: angry;
    std::vector<int> task_rejected_times_;
    std::unordered_map<int, std::unordered_map<int, std::unordered_set<int>>>
        task_id_pid_tids_;  // task_id : pid : {tid1, tid2...}
    std::list<int> angry_list_; // storing task_ids that wait too long
};

} // namespace husky

