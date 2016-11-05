
#include "core/worker/main_loop.hpp"

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

    husky::MainLoop main_loop(std::move(worker_info),
            std::move(master_connector));
    main_loop.serve();
}
