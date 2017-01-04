#include "worker/engine.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();

    // round 1: 
    // worker_nums: 2,2,3
    auto task1 = TaskFactory::Get().CreateTask<Task>(1, 2);
    engine.AddTask(task1, [](const Info& info) { base::log_msg(std::to_string(info.get_task_id()) + " is running"); });

    auto task2 = TaskFactory::Get().CreateTask<Task>(1, 2);
    engine.AddTask(task2, [](const Info& info) { base::log_msg(std::to_string(info.get_task_id()) + " is running"); });

    auto task3 = TaskFactory::Get().CreateTask<Task>(1, 3);
    engine.AddTask(task3, [](const Info& info) { base::log_msg(std::to_string(info.get_task_id()) + " is running"); });
    engine.Submit();

    // round 2
    // worker_nums: 2, 2: multiple epochs
    auto task4 = TaskFactory::Get().CreateTask<Task>(4, 2);
    engine.AddTask(task4, [](const Info& info) { base::log_msg(std::to_string(info.get_task_id()) + " is running"); });

    auto task5 = TaskFactory::Get().CreateTask<Task>(4, 2);
    engine.AddTask(task5, [](const Info& info) { base::log_msg(std::to_string(info.get_task_id()) + " is running"); });
    engine.Submit();

    // round 3
    // worker_nums: 2, 2: hogwild!
    auto task6 = TaskFactory::Get().CreateTask<HogwildTask>(2, 4);
    engine.AddTask(task6, [](const Info& info) { base::log_msg(std::to_string(info.get_task_id()) + " is running"); });

    auto task7 = TaskFactory::Get().CreateTask<Task>(2, 2);
    engine.AddTask(task7, [](const Info& info) { base::log_msg(std::to_string(info.get_task_id()) + " is running"); });
    engine.Submit();

    engine.Exit();
}
