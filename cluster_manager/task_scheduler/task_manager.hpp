#pragma once

#include <list>
#include <unordered_set> 
#include <unordered_map> 

#include "cluster_manager/task_scheduler/available_workers.hpp"
#include "cluster_manager/task_scheduler/history_manager.hpp" 
#include "core/color.hpp"

namespace husky {
// considering both the history and priority

class TaskManager {
   public:
    TaskManager() {}
    TaskManager(const TaskManager&) = delete;

    // add tasks into the scheduling pool    
    void add_tasks(const std::vector<std::shared_ptr<Task>>& tasks);
    

    bool exist_angry_tasks();

    // check if all the tasks have fininshed all their epochs
    bool is_finished();

    // untrack the thread and if no threads running for this task
    // set this task to be ready or finished
    void finish_thread(int task_id, int pid, int global_thread_id);

    // return task ids orderred by the decreasing priority of tasks
    std::vector<int> order_by_priority();

    // return process ids that the task least frequently visited
    std::vector<int> get_preferred_proc(int task_id);

    // set task status to running and decrease its priority
    void suc_sched(int task_id);

    // increase its priority and possibly add it into angry list
    void fail_sched(int task_id);

    // update the history of this task and track the threads using task_id_pid_tids_
    void record_and_track(int task_id, std::vector<std::pair<int, int>>& pid_tid);

    int get_num_tasks();

    std::shared_ptr<Task> get_task_by_id(int task_id);

    int get_task_priority(int task_id);

    int get_task_status(int task_id);

    int get_task_rej_times(int task_id);
    
    std::unordered_set<int> get_tracking_threads(int task_id, int proc_id);

    std::list<int>::const_iterator angry_list_begin() {return angry_list_.begin(); }

    std::list<int>::const_iterator angry_list_end() {return angry_list_.end(); }

   private: 
    // check whether a task is angry
    bool in_angry_list(int task_id);

    // remove from angry task list
    void erase_angry_task(int task_id);
    
   private:
    // the threshold to add a task in angry list
    int angry_threshold_ = 5;

    int num_tasks_ = 0;

    // store all the tasks indexed by task_id
    std::vector<std::shared_ptr<Task>> tasks_;

    // vector of <task_id, priority> 
    std::vector<std::pair<int, int>> task_priority_; 

    // 0: ready; 1: running; 2: finished; 3: angry; indexed by task_id
    std::vector<int> task_status_;

    // tracking how many times a task has been rejected, indexed by task_id
    std::vector<int> task_rejected_times_;

    // task_id : pid : {tid1, tid2...}
    // tracking the running threads assigned to a task
    std::unordered_map<int, std::unordered_map<int, std::unordered_set<int>>>
        task_id_pid_tids_;

    // storing task_ids that wait too long
    std::list<int> angry_list_;
};

} // namespace husky

