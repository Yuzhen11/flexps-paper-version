#include "cluster_manager/task_scheduler/task_manager.hpp"

#include <vector>
#include <memory>
#include <iostream>

#include "gtest/gtest.h"

#include "core/task.hpp"
#include "cluster_manager/task_scheduler/history_manager.hpp"

namespace husky {
namespace {

class TestTaskManager: public testing::Test {
   public:
    TestTaskManager() {}
    ~TestTaskManager() {}
   protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(TestTaskManager, TestAddTasks) {
    TaskManager task_manager;
    int id = 0;
    int total_epoch = 1;
    int num_workers = 1; 
    std::shared_ptr<Task> task1(new Task(id, total_epoch, num_workers, Task::Type::BasicTaskType));
    std::shared_ptr<Task> task2(new Task(id+1, total_epoch+1, num_workers+1, Task::Type::BasicTaskType));
    std::vector<std::shared_ptr<Task>> tasks;
    tasks.push_back(task1);
    tasks.push_back(task2);
    task_manager.add_tasks(tasks);

    EXPECT_EQ(task_manager.get_num_tasks(), 2);
    EXPECT_EQ(task_manager.get_task_priority(0) , 0);
    EXPECT_EQ(task_manager.get_task_priority(1) , 0);
    EXPECT_EQ(task_manager.get_task_status(0) , 0);
    EXPECT_EQ(task_manager.get_task_status(1) , 0);
    EXPECT_EQ(task_manager.get_task_rej_times(0) , 0);
    EXPECT_EQ(task_manager.get_task_rej_times(1) , 0);
    
    std::shared_ptr<Task> task3(new Task(id+2, total_epoch+2, num_workers+2, Task::Type::BasicTaskType));
    std::shared_ptr<Task> task4(new Task(id+3, total_epoch+3, num_workers+3, Task::Type::BasicTaskType));
    std::vector<std::shared_ptr<Task>> tasks1;
    tasks1.push_back(task3);
    tasks1.push_back(task4);
    task_manager.add_tasks(tasks1);

    EXPECT_EQ(task_manager.get_num_tasks() , 4);
    EXPECT_EQ(task_manager.get_task_priority(0) , 0);
    EXPECT_EQ(task_manager.get_task_priority(3) , 0);
    EXPECT_EQ(task_manager.get_task_status(0) , 0);
    EXPECT_EQ(task_manager.get_task_status(3) , 0);
    EXPECT_EQ(task_manager.get_task_rej_times(0) , 0);
    EXPECT_EQ(task_manager.get_task_rej_times(3) , 0);
}

TEST_F(TestTaskManager, TestFailSched) {
    TaskManager task_manager;
    int id = 0;
    int total_epoch = 1;
    int num_workers = 1; 
    std::shared_ptr<Task> task1(new Task(id, total_epoch, num_workers, Task::Type::BasicTaskType));
    std::shared_ptr<Task> task2(new Task(id+1, total_epoch+1, num_workers+1, Task::Type::BasicTaskType));
    std::vector<std::shared_ptr<Task>> tasks;
    tasks.push_back(task1);
    tasks.push_back(task2);

    task_manager.add_tasks(tasks);
    task_manager.fail_sched(0);
    task_manager.fail_sched(0);
    task_manager.fail_sched(1);
    EXPECT_EQ(task_manager.get_task_priority(0) , 2);
    EXPECT_EQ(task_manager.get_task_priority(1) , 1);
}

TEST_F(TestTaskManager, TestSucFailSched) {
    TaskManager task_manager;
    int id = 0;
    int total_epoch = 1;
    int num_workers = 1; 
    std::shared_ptr<Task> task1(new Task(id, total_epoch, num_workers, Task::Type::BasicTaskType));
    std::shared_ptr<Task> task2(new Task(id+1, total_epoch+1, num_workers+1, Task::Type::BasicTaskType));
    std::shared_ptr<Task> task3(new Task(id+2, total_epoch+2, num_workers+2, Task::Type::BasicTaskType));
    std::vector<std::shared_ptr<Task>> tasks;
    tasks.push_back(task1);
    tasks.push_back(task2);
    tasks.push_back(task3);

    task_manager.add_tasks(tasks);
    task_manager.fail_sched(0);
    task_manager.fail_sched(0);
    task_manager.fail_sched(1);
    task_manager.fail_sched(2);
    EXPECT_EQ(task_manager.get_task_priority(0) , 2);
    EXPECT_EQ(task_manager.get_task_priority(1) , 1);
    EXPECT_EQ(task_manager.get_task_priority(2) , 1);

    task_manager.suc_sched(0);
    task_manager.suc_sched(1);
    task_manager.suc_sched(2);
    EXPECT_EQ(task_manager.get_task_priority(0) , 0);
    EXPECT_EQ(task_manager.get_task_priority(1) , 0);
    EXPECT_EQ(task_manager.get_task_priority(2) , 0);
}

TEST_F(TestTaskManager, TestOrderByPriority) {
    TaskManager task_manager;
    int id = 0;
    int total_epoch = 1;
    int num_workers = 1; 
    std::shared_ptr<Task> task1(new Task(id, total_epoch, num_workers, Task::Type::BasicTaskType));
    std::shared_ptr<Task> task2(new Task(id+1, total_epoch+1, num_workers+1, Task::Type::BasicTaskType));
    std::shared_ptr<Task> task3(new Task(id+2, total_epoch+2, num_workers+2, Task::Type::BasicTaskType));
    std::vector<std::shared_ptr<Task>> tasks;
    tasks.push_back(task1);
    tasks.push_back(task2);
    tasks.push_back(task3);

    task_manager.add_tasks(tasks);
    task_manager.fail_sched(0);
    task_manager.fail_sched(1);
    task_manager.fail_sched(1);
    task_manager.fail_sched(1);
    task_manager.fail_sched(2);
    task_manager.fail_sched(2);

    std::vector<int> result = task_manager.order_by_priority();
    EXPECT_EQ(result[0] , 1);
    EXPECT_EQ(result[1] , 2);
    EXPECT_EQ(result[2] , 0);
}

TEST_F(TestTaskManager, TestRecordAndTrack) {
    HistoryManager::get().clear_history();
    HistoryManager::get().start(2);
    TaskManager task_manager;
    int id = 0;
    int total_epoch = 1;
    int num_workers = 1; 
    std::shared_ptr<Task> task1(new Task(id, total_epoch, num_workers, Task::Type::BasicTaskType));
    std::shared_ptr<Task> task2(new Task(id+1, total_epoch+1, num_workers+1, Task::Type::BasicTaskType));
    std::vector<std::shared_ptr<Task>> tasks;
    tasks.push_back(task1);
    tasks.push_back(task2);

    task_manager.add_tasks(tasks);
    std::vector<std::pair<int, int>> pid_tids;
    task_manager.record_and_track(1, pid_tids); 

    pid_tids.push_back({1, 2});
    pid_tids.push_back({1, 3});
    pid_tids.push_back({1, 4});
    task_manager.record_and_track(0, pid_tids); 
    const auto& set0 = task_manager.get_tracking_threads(0, 1);// task_id, process_id
    const auto& set1 = task_manager.get_tracking_threads(1, 1);
    EXPECT_EQ(set0.size(), 3);
    EXPECT_EQ(set1.size(), 0);
}

TEST_F(TestTaskManager, TestGetPreferredProc) {
    HistoryManager::get().clear_history();
    HistoryManager::get().start(10);
    TaskManager task_manager;
    int id = 0;
    int total_epoch = 1;
    int num_workers = 1; 
    std::shared_ptr<Task> task1(new Task(id, total_epoch, num_workers, Task::Type::BasicTaskType));
    std::shared_ptr<Task> task2(new Task(id+1, total_epoch+1, num_workers+2, Task::Type::BasicTaskType));
    std::shared_ptr<Task> task3(new Task(id+2, total_epoch+2, num_workers+5, Task::Type::BasicTaskType));
    std::vector<std::shared_ptr<Task>> tasks;
    tasks.push_back(task1);
    tasks.push_back(task2);
    tasks.push_back(task3);

    task_manager.add_tasks(tasks);
    std::vector<std::pair<int, int>> pid_tids;
    task_manager.record_and_track(0, pid_tids); // track task 0 and empty pid_tids

    pid_tids.push_back({1, 11}); // proc_id thread id
    pid_tids.push_back({2, 21});
    pid_tids.push_back({3, 31});
    task_manager.record_and_track(1, pid_tids); 

    pid_tids.push_back({0, 1}); // proc_id thread id
    pid_tids.push_back({5, 51});
    pid_tids.push_back({6, 61});
    task_manager.record_and_track(2, pid_tids); 
    const auto& pre_0 = task_manager.get_preferred_proc(0);
    EXPECT_EQ(pre_0.size(), 10);// no preference

    const auto& pre_1 = task_manager.get_preferred_proc(1);
    EXPECT_EQ(pre_1[0], 0);
    EXPECT_EQ(pre_1.size(), 7);

    const auto& pre_2 = task_manager.get_preferred_proc(2);
    EXPECT_EQ(pre_2[1], 7);
    EXPECT_EQ(pre_2.size(), 4);
}

TEST_F(TestTaskManager, TestFinishThread) {
    HistoryManager::get().clear_history();
    HistoryManager::get().start(10);
    TaskManager task_manager;
    int id = 0;
    int total_epoch = 1;
    int num_workers = 1; 
    std::shared_ptr<Task> task1(new Task(id, total_epoch, num_workers, Task::Type::BasicTaskType));
    std::vector<std::shared_ptr<Task>> tasks;
    tasks.push_back(task1);

    task_manager.add_tasks(tasks);
    std::vector<std::pair<int, int>> pid_tids;
    pid_tids.push_back({1, 11}); // proc_id thread id
    pid_tids.push_back({1, 21});
    task_manager.record_and_track(0, pid_tids); // track task 0

    task_manager.finish_thread(0, 1, 11);
    const auto& set0 = task_manager.get_tracking_threads(0, 1);// task_id, process_id
    EXPECT_EQ(set0.size() , 1);
    EXPECT_TRUE(set0.find(21) != set0.end());

    task_manager.finish_thread(0, 1, 21);
    const auto& set1 = task_manager.get_tracking_threads(0, 1);// task_id, process_id
    EXPECT_EQ(set1.find(21) , set0.end());
    EXPECT_EQ(set1.size() , 0);

    EXPECT_EQ(task_manager.get_task_status(0), 2);
    EXPECT_TRUE(task_manager.is_finished());
}

TEST_F(TestTaskManager, TestAngryListFunction) {
    TaskManager task_manager;
    int id = 0, id2 = 1;
    int total_epoch = 1;
    int num_workers = 1; 
    std::shared_ptr<Task> task1(new Task(id, total_epoch, num_workers, Task::Type::BasicTaskType));
    std::shared_ptr<Task> task2(new Task(id2, total_epoch+1, num_workers+1, Task::Type::BasicTaskType));
    std::vector<std::shared_ptr<Task>> tasks;
    tasks.push_back(task1);
    tasks.push_back(task2);
    task_manager.add_tasks(tasks);
    task_manager.fail_sched(id);
    task_manager.fail_sched(id);
    task_manager.fail_sched(id);
    task_manager.fail_sched(id);
    task_manager.fail_sched(id);
    task_manager.fail_sched(id);

    task_manager.fail_sched(id2);
    task_manager.fail_sched(id2);
    task_manager.fail_sched(id2);
    task_manager.fail_sched(id2);
    task_manager.fail_sched(id2);

    auto list_begin = task_manager.angry_list_begin();
    EXPECT_EQ(*list_begin , id);

    task_manager.suc_sched(id);
    task_manager.suc_sched(id2);
    EXPECT_TRUE(!(task_manager.exist_angry_tasks()));
}

}  // namespace
} // namespace husky
