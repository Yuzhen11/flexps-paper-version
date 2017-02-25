#include "cluster_manager/task_scheduler/priority_task_scheduler.hpp"

#include <vector>
#include <memory>
#include <utility>
#include "gtest/gtest.h"

#include "cluster_manager/task_scheduler/history_manager.hpp"

namespace husky {
namespace {

class TestPriorityTaskScheduler: public testing::Test {
   public:
    TestPriorityTaskScheduler() {
        int num_local_workers = 10;
        for (int i = 0; i<num_process; i++) {
            int process_id = i;
            for (int j = 0; j<num_local_workers;j++) {
                int global_worker_id = 10*i + j;
                int local_worker_id = j;
                worker_info.add_worker(process_id, global_worker_id, local_worker_id);
            }
        }

        int id = 0;
        int total_epoch = 1;
        int num_workers = 1; 
        int num_tasks = 3;
        for (int i=0; i<num_tasks; i++) {
            std::shared_ptr<Task> task_ptr(new Task(id+i, total_epoch+i, num_workers+i, Task::Type::MLTaskType));
            tasks.push_back(std::move(task_ptr));
        }
    }
    ~TestPriorityTaskScheduler() {}
    
    int num_process = 10;
    std::vector<std::shared_ptr<Task>> tasks;
    WorkerInfo worker_info; 

   protected:
    void SetUp() {}
    void TearDown() {}
};


TEST_F(TestPriorityTaskScheduler, TestCreate) {
    PriorityTaskScheduler pts(worker_info);
}

TEST_F(TestPriorityTaskScheduler, TestInitTasks) {
    PriorityTaskScheduler pts(worker_info);
    pts.init_tasks(tasks);
}

TEST_F(TestPriorityTaskScheduler, TestExtractInstancesSingle) {
    HistoryManager::get().clear_history();
    for (auto& task_ptr : tasks) {
        task_ptr->set_hint("single");
    }
    HistoryManager::get().start(num_process);
    PriorityTaskScheduler pts(worker_info);
    pts.init_tasks(tasks);
    pts.extract_instances();
}

TEST_F(TestPriorityTaskScheduler, TestExtractInstancesPS) {
    HistoryManager::get().clear_history();
    for (auto& task_ptr : tasks) {
        task_ptr->set_hint("PS:BSP");
    }
    HistoryManager::get().start(num_process);
    PriorityTaskScheduler pts(worker_info);
    pts.init_tasks(tasks);
    pts.extract_instances();
}

} // namespace
} // namespace husky
