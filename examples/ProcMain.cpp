
#include "core/main_loop.hpp"

using namespace husky;

int main() {
    std::string main_loop_listen_port = "12345";

    // worker info
    WorkerInfo worker_info;
    worker_info.add_proc(0, "proj10");
    worker_info.add_worker(0,0,0);
    worker_info.add_worker(0,1,1);
    worker_info.set_num_processes(1);
    worker_info.set_num_workers(2);
    worker_info.set_proc_id(0);

    zmq::context_t context;
    husky::MainLoop main_loop(std::move(worker_info), context, main_loop_listen_port);
    main_loop.serve();
}
