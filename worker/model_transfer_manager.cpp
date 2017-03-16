#include "worker/model_transfer_manager.hpp"

namespace husky {

ModelTransferManager::ModelTransferManager(const WorkerInfo& worker_info,
        MailboxEventLoop* const el,
        zmq::context_t* context):
    context_(context),
    worker_info_(worker_info) {
    Serve(el);
}
ModelTransferManager::~ModelTransferManager() {
    SendHalt();
}
void ModelTransferManager::Serve(MailboxEventLoop* const el) {

    // Set the mailbox
    // The kvstore will use mailboxes [num_workers,2*num_workers + num_processes_*num_servers_per_process)
    // The assumption here is that num_servers_per_process will not be larger than 20!!!
    // The following mailboxes [2*num_workers + num_processes_*20, 2*num_workers + num_processes_*20 + num_processes_)
    int max_servers_per_process = 20;  // Magic number here
    int num_processes = worker_info_.get_num_processes();
    int num_workers = worker_info_.get_num_workers();
    int base = 2*num_workers + num_processes * max_servers_per_process;
    for (int i = 0; i < num_processes; ++ i) {
        int tid = base + i;
        if (i != worker_info_.get_process_id()) {
            el->register_peer_thread(i, tid);
        } else {
            auto* mailbox = new husky::LocalMailbox(context_);
            mailbox->set_thread_id(tid);
            el->register_mailbox(*mailbox);
            mailbox_.reset(mailbox);
        }
    }

    // recv socket
    recv_socket_.reset(new zmq::socket_t(*context_, ZMQ_PULL));
    recv_socket_->bind(kBindAddr);
    recv_thread_ = std::thread(&ModelTransferManager::Main, this);

    // send socket
    send_socket_.reset(new zmq::socket_t(*context_, ZMQ_PUSH));
    send_socket_->connect(kBindAddr);
}

/*
 * Worker uses SendTask to send the task to ModelTransferManager's event-loop
 */
void ModelTransferManager::SendTask(int dst, int model_id) {
    zmq_sendmore_int32(send_socket_.get(), kCmdTask);
    zmq_sendmore_int32(send_socket_.get(), dst);
    zmq_send_int32(send_socket_.get(), model_id);
}

/*
 * Worker uses SendHalt to halt the ModelTransferManager's event-loop
 */
void ModelTransferManager::SendHalt() {
    zmq_send_int32(send_socket_.get(), kCmdHalt);
    recv_thread_.join();
    // close sockets
    recv_socket_->close();
    recv_socket_.reset(nullptr);
    send_socket_->close();
    send_socket_.reset(nullptr);
    mailbox_.reset(nullptr);
}

void ModelTransferManager::Main() {
    while(true) {
        int type = zmq_recv_int32(recv_socket_.get());
        switch (type) {
        case kCmdTask: {
            int dst = zmq_recv_int32(recv_socket_.get());
            int model_id = zmq_recv_int32(recv_socket_.get());
            base::BinStream bin = ModelTransferStore::Get().Pop(model_id);
            husky::LOG_I<< RED("Sending model: dst: "+std::to_string(dst)+" model_id: "+std::to_string(model_id));
            mailbox_->send(dst, 0, 0, bin);
            break;
        }
        case kCmdHalt: {
            return;
        }
        default: {
            throw base::HuskyException("Unknown type in ModelTransferManager: "+std::to_string(type));
        }
        }
    }
}

}  // namespace husky
