
#include <chrono>

#include "core/worker/worker.hpp"

using namespace husky;

int main() {
    std::string bind_addr = "tcp://*:12345";  // for main loop
    std::string master_addr = "tcp://proj10:45123";
    std::string host_name = "proj10";

    // worker info
    WorkerInfo worker_info;
    worker_info.add_proc(0, "proj10");
    worker_info.add_worker(0,0,0);
    worker_info.add_worker(0,1,1);
    worker_info.set_num_processes(1);
    worker_info.set_num_workers(2);
    worker_info.set_proc_id(0);

    // master connector
    zmq::context_t context;
    MasterConnector master_connector(context, bind_addr, master_addr, host_name);


    // tasks
    Task task1(0,1,2);  // id: 0, total_epoch: 1, num_workers: 2

    husky::Worker worker(std::move(worker_info),
            std::move(master_connector));
    // add task
    worker.add_task(task1, [](){
        std::this_thread::sleep_for(std::chrono::seconds(1));
        base::log_msg("task1 is running");
    });
    worker.send_tasks_to_master();
    worker.main_loop();
}
