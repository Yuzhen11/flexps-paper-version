#include "cluster_manager/task_scheduler/sequential_task_scheduler.hpp"

#include <vector>
#include <memory>
#include <utility>
#include "gtest/gtest.h"

#include "cluster_manager/task_scheduler/history_manager.hpp"
#include "core/constants.hpp"

namespace husky {
namespace {

class TestSequentialTaskScheduler: public testing::Test {
   public:
    TestSequentialTaskScheduler() {
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
        // insert three tasks with the same id
            std::shared_ptr<Task> task_ptr(new Task(id, total_epoch+i, num_workers+i, Task::Type::MLTaskType));
            tasks.push_back(std::move(task_ptr));
        }
    }
    ~TestSequentialTaskScheduler() {}
    
    int num_process = 10;
    std::vector<std::shared_ptr<Task>> tasks;
    WorkerInfo worker_info; 

   protected:
    void SetUp() {}
    void TearDown() {}
};


TEST_F(TestSequentialTaskScheduler, TestCreate) {
    SequentialTaskScheduler sts(worker_info);
}

TEST_F(TestSequentialTaskScheduler, TestInitTasks) {
    SequentialTaskScheduler sts(worker_info);
    sts.init_tasks(tasks);
}

TEST_F(TestSequentialTaskScheduler, TestExtractInstancesSingleProc) {
    HistoryManager::get().clear_history();

    for (auto& task_ptr : tasks) {
        if (task_ptr->get_num_workers() == 1) {
            task_ptr->set_local();
        }
        else {
            // SPMT share the same scheduling startegy
            task_ptr->set_local();
        }
    }
    HistoryManager::get().start(num_process);
    SequentialTaskScheduler sts(worker_info);
    sts.init_tasks(tasks);
    sts.extract_instances();
}

TEST_F(TestSequentialTaskScheduler, TestExtractInstancesPS) {
    HistoryManager::get().clear_history();
    HistoryManager::get().start(num_process);
    SequentialTaskScheduler sts(worker_info);
    sts.init_tasks(tasks);
    sts.extract_instances();
}

} // namespace
} // namespace husky
