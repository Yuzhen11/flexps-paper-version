#include <chrono>

#include "worker/task_factory.hpp"
#include "worker/worker.hpp"

using namespace husky;

int main() {
    std::string bind_addr = "tcp://*:12345";  // for main loop
    std::string cluster_manager_addr = "tcp://proj10:45123";
    std::string host_name = "proj10";

    // worker info
    WorkerInfo worker_info;
    // worker_info.add_proc(0, "proj10");
    worker_info.add_worker(0, 0, 0);
    worker_info.add_worker(0, 1, 1);
    worker_info.set_process_id(0);

    // cluster_manager connector
    zmq::context_t context;
    ClusterManagerConnector cluster_manager_connector(&context, bind_addr, cluster_manager_addr, host_name);

    // create worker
    husky::Worker worker(std::move(worker_info), std::move(cluster_manager_connector));

    // add tasks
    auto task1 =
        TaskFactory::Get().CreateTask<Task>(1, 2);  // id: 0, total_epoch: 1, num_workers: 2
    worker.add_task(task1, [](const Info& info) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        base::log_msg("task1 is running");
    });

    auto task2 =
        TaskFactory::Get().CreateTask<Task>(2, 1);  // id: 1, total_epoch: 2, num_workers: 1
    worker.add_task(task2, [](const Info& info) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        base::log_msg("task2 is running");
    });

    worker.send_tasks_to_cluster_manager();
    worker.main_loop();
}
