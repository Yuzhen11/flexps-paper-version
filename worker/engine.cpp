#include "worker/engine.hpp"

namespace husky {

void Engine::Submit() {
    worker->send_tasks_to_cluster_manager();
    worker->main_loop();
}
void Engine::Exit() {
    StopWorker();
    StopCoordinator();
}

Engine::Engine() {
    StartWorker();
    StartCoordinator();
}
void Engine::StartWorker() {
    std::string bind_addr = "tcp://*:" + Context::get_param("worker_port");
    std::string cluster_manager_addr =
        "tcp://" + Context::get_param("cluster_manager_host") + ":" + Context::get_param("cluster_manager_port");
    std::string host_name = Context::get_param("hostname");

    // worker info
    auto& worker_info = Context::get_worker_info();

    // cluster_manager connector
    ClusterManagerConnector cluster_manager_connector(Context::get_zmq_context(), bind_addr, cluster_manager_addr,
                                                      host_name);
    // Create mailboxes
    Context::create_mailbox_env();

    // Create ModelTransferManager
    model_transfer_manager.reset(new ModelTransferManager(worker_info, Context::get_mailbox_event_loop(), Context::get_zmq_context()));

    // create worker
    worker.reset(new Worker(worker_info, model_transfer_manager.get(), std::move(cluster_manager_connector)));
}

void Engine::StartCoordinator() { 
    Context::get_coordinator()->serve(); 
}
void Engine::StopWorker() {
    worker->send_exit(); 
}

void Engine::StopCoordinator() {
    for (auto tid : Context::get_worker_info().get_local_tids()) {
        base::BinStream finish_signal;
        finish_signal << Context::get_param("hostname") << tid;
        Context::get_coordinator()->notify_master(finish_signal, TYPE_EXIT);
    }
}

}  // namespace husky
