#pragma once

#include <cstdlib>
#include <ctime>
#include <queue>
#include <unordered_map>
#include "core/instance.hpp"
#include "core/task.hpp"

namespace husky {

/*
 * TaskScheduler interface
 */
class TaskScheduler {
   public:
    /*
     * Constructor: construct TaskScheduler with a worker_info
     */
    TaskScheduler(WorkerInfo& worker_info_) : worker_info(worker_info_) {}
    /*
     * Initialize the TaskScheduler with a vector of Task
     */
    virtual void init_tasks(const std::vector<std::shared_ptr<Task>>&) = 0;
    /*
     * Invoke the function when some threads are finished
     */
    virtual void finish_thread(int instance_id, int global_thread_id) = 0;
    /*
     * Return a vector of new instances if there is any
     */
    virtual std::vector<std::shared_ptr<Instance>> extract_instances() = 0;
    /*
     * Whether all the tasks are done
     */
    virtual bool is_finished() = 0;

    virtual ~TaskScheduler(){};

   protected:
    WorkerInfo& worker_info;
};

}  // namespace husky
