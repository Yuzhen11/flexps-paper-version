#include "core/worker/worker.hpp"
#include "core/common/context.hpp"

using namespace husky;

int main(int argc, char** argv) {
    Context::init_global();
    bool rt = Context::get_config()->init_with_args(argc, argv, {});
    if (!rt) return 1;

    std::string bind_addr = "tcp://*:"+std::to_string(Context::get_config()->get_worker_port());
    std::string master_addr = "tcp://"+Context::get_config()->get_master_host()+":"+std::to_string(Context::get_config()->get_master_port());
    std::string host_name = Context::get_param("hostname");

    // worker info
    WorkerInfo worker_info = *Context::get_worker_info();

    // master connector
    MasterConnector master_connector(Context::get_zmq_context(), bind_addr, master_addr, host_name);

    // create worker
    husky::Worker worker(std::move(worker_info),
            std::move(master_connector));

    // add tasks
    Task task1(0,1,2);  // id: 0, total_epoch: 1, num_workers: 2
    worker.add_task(task1, [](){
        std::this_thread::sleep_for(std::chrono::seconds(1));
        base::log_msg("task1 is running");
    });

    Task task2(1,2,1);  // id: 1, total_epoch: 2, num_workers: 1
    worker.add_task(task2, [](){
        std::this_thread::sleep_for(std::chrono::seconds(1));
        base::log_msg("task2 is running");
    });

    worker.send_tasks_to_master();
    worker.main_loop();
}
