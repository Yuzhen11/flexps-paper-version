#include "master/master.hpp"
#include "husky/core/context.hpp"
#include "husky/core/job_runner.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port"});
    if (!rt) return 1;

    std::string bind_addr = "tcp://*:"+std::to_string(Context::get_config().get_master_port());
    // assume that all remote worker port are the same
    std::string remote_port = Context::get_param("worker_port");

    // worker info
    WorkerInfo worker_info = Context::get_worker_info();

    // master connection
    MasterConnection master_connection(*Context::get_zmq_context(), bind_addr);

    auto& procs = worker_info.get_hostnames();
    for (int i = 0; i < procs.size(); ++ i) {
        master_connection.add_proc(i, "tcp://"+procs[i]+":"+remote_port);
    }


    Master master(std::move(worker_info),
            std::move(master_connection));
    // master.test_connection();
    master.master_loop();
}
