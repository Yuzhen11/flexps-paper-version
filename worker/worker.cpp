#include "worker/worker.hpp"

namespace husky {

Worker::Worker(const WorkerInfo& worker_info_, ModelTransferManager* model_transfer_manager, ClusterManagerConnector&& cluster_manager_connector_)
    : worker_info(worker_info_),
      model_transfer_manager_(model_transfer_manager),
      cluster_manager_connector(std::move(cluster_manager_connector_)),
      instance_runner(worker_info, cluster_manager_connector, task_store) {}

void Worker::send_tasks_to_cluster_manager() {
    // Only Proc 0 need to send tasks to cluster_manager
    if (worker_info.get_process_id() == 0) {
        base::BinStream bin;
        auto& task_map = task_store.get_task_map();
        auto& buffered_tasks = task_store.get_buffered_tasks();
        // send out buffered_tasks
        bin << buffered_tasks.size();
        for (auto id : buffered_tasks) {
            auto& task = task_map[id].first;
            bin << task->get_type();  // push the task type first
            task->serialize(bin);     // push the task
        }
        auto& socket = cluster_manager_connector.get_send_socket();
        zmq_sendmore_int32(&socket, constants::kClusterManagerInit);
        zmq_send_binstream(&socket, bin);
        husky::LOG_I << GREEN("[Worker]: " + std::to_string(buffered_tasks.size()) + " tasks sent");
        // clear buffered tasks
        task_store.clear_buffered_tasks();
    }
}

/*
 * send exit signal to cluster_manager, stop the cluster_manager
 * normally it's the last statement in worker
 */
void Worker::send_exit() {
    // stop the model_transfer_manager_
    // model_transfer_manager_->SendHalt();
    if (worker_info.get_process_id() == 0) {
        auto& socket = cluster_manager_connector.get_send_socket();
        zmq_send_int32(&socket, constants::kClusterManagerExit);
    }
}

void Worker::send_thread_finished(int instance_id, int thread_id) {
    int global_thread_id = worker_info.local_to_global_id(thread_id);
    base::BinStream bin;
    bin << instance_id << global_thread_id;
    auto& socket = cluster_manager_connector.get_send_socket();
    zmq_sendmore_int32(&socket, constants::kClusterManagerThreadFinished);
    zmq_send_binstream(&socket, bin);  // {instance_id, global_thread_id}
}

void Worker::main_loop() {
    auto& socket = cluster_manager_connector.get_recv_socket();
    auto& send_socket = cluster_manager_connector.get_send_socket();
    while (true) {
        int type = zmq_recv_int32(&socket);
        // husky::LOG_I << "[Worker]: Msg Type: " + std::to_string(type);
        if (type == constants::kTaskType) {
            auto bin = zmq_recv_binstream(&socket);
            std::shared_ptr<Instance> instance(new Instance);
            instance->deserialize(bin);
            // instance->show_instance(worker_info.get_process_id());
            instance_runner.run_instance(instance);
        } else if (type == constants::kThreadFinished) {
            int instance_id = zmq_recv_int32(&socket);
            int thread_id = zmq_recv_int32(&socket);
            instance_runner.finish_thread(instance_id, thread_id);
            // tell master
            send_thread_finished(instance_id, thread_id);
            bool is_instance_done = instance_runner.is_instance_done(instance_id);
            if (is_instance_done) {
                husky::LOG_I << GREEN("[Worker]: Instance id:" + std::to_string(instance_id) +
                                      " finished on Proc:" + std::to_string(worker_info.get_process_id()));
                instance_runner.remove_instance(instance_id);
            }
        } else if (type == constants::kClusterManagerDirectTransferModel) {
            int dst = zmq_recv_int32(&socket);
            int model_id = zmq_recv_int32(&socket);
            model_transfer_manager_->SendTask(dst, model_id);
        } else if (type == constants::kClusterManagerFinished) {
            husky::LOG_I << GREEN("[Worker]: Tasks finished");
            break;
        } else {
            throw base::HuskyException("[Worker] Worker Loop recv type error, type is: " + std::to_string(type));
        }
    }
}

}  // namespace husky
