#include "worker/engine.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();

    // round 1: 
    // worker_nums: 2,2,3
    auto task1 = TaskFactory::Get().create_task(Task::Type::BasicTaskType, 1, 2);
    engine.AddTask(std::move(task1), [](const Info& info) { base::log_msg(std::to_string(info.get_task_id()) + " is running"); });

    auto task2 = TaskFactory::Get().create_task(Task::Type::BasicTaskType, 1, 2);
    engine.AddTask(std::move(task2), [](const Info& info) { base::log_msg(std::to_string(info.get_task_id()) + " is running"); });

    auto task3 = TaskFactory::Get().create_task(Task::Type::BasicTaskType, 1, 3);
    engine.AddTask(std::move(task3), [](const Info& info) { base::log_msg(std::to_string(info.get_task_id()) + " is running"); });
    engine.Submit();

    // round 2
    // worker_nums: 2, 2: multiple epochs
    auto task4 = TaskFactory::Get().create_task(Task::Type::BasicTaskType, 4, 2);
    engine.AddTask(std::move(task4), [](const Info& info) { base::log_msg(std::to_string(info.get_task_id()) + " is running"); });

    auto task5 = TaskFactory::Get().create_task(Task::Type::BasicTaskType, 4, 2);
    engine.AddTask(std::move(task5), [](const Info& info) { base::log_msg(std::to_string(info.get_task_id()) + " is running"); });
    engine.Submit();

    // round 3
    // worker_nums: 2, 2: hogwild!
    auto task6 = TaskFactory::Get().create_task(Task::Type::HogwildTaskType, 2, 4);
    engine.AddTask(std::move(task6), [](const Info& info) { base::log_msg(std::to_string(info.get_task_id()) + " is running"); });

    auto task7 = TaskFactory::Get().create_task(Task::Type::BasicTaskType, 2, 2);
    engine.AddTask(std::move(task7), [](const Info& info) { base::log_msg(std::to_string(info.get_task_id()) + " is running"); });
    engine.Submit();

    engine.Exit();
}
