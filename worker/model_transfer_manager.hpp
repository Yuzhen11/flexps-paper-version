#pragma once

#include <thread>

#include "husky/core/zmq_helpers.hpp"
#include "husky/core/mailbox.hpp"
#include "husky/base/exception.hpp"
#include "worker/model_transfer_store.hpp"

namespace husky {

/*
 * ModelTransferManager
 *
 * Receive signal from Worker to directly transfer the model to the process
 * where next instance will run on.
 *
 */
class ModelTransferManager {
   public:
    static const int kCmdTask = 0;
    static const int kCmdHalt= 1;
    static constexpr const char* kBindAddr = "inproc://model_tranfer_manager";

    ModelTransferManager(const WorkerInfo& worker_info,
            MailboxEventLoop* const el,
            zmq::context_t* context):
        context_(context),
        worker_info_(worker_info) {
        Serve(el);
    }
    ModelTransferManager(const ModelTransferManager&) = delete;
    ModelTransferManager& operator=(const ModelTransferManager&) = delete;

    void Serve(MailboxEventLoop* const el) {

        // Set the mailbox
        // The kvstore will use mailboxes [num_workers,2*num_workers + num_processes_*num_servers_per_process)
        // The assumption here is that num_servers_per_process will not be larger than 10!!!
        // The following mailboxes [2*num_workers + num_processes_*10, 2*num_workers + num_processes_*10 + num_processes_)
        int max_servers_per_process = 10;  // Magic number here
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
    void SendTask(int dst, int model_id) {
        zmq_sendmore_int32(send_socket_.get(), kCmdTask);
        zmq_sendmore_int32(send_socket_.get(), dst);
        zmq_send_int32(send_socket_.get(), model_id);
    }

    /*
     * Worker uses SendHalt to halt the ModelTransferManager's event-loop
     */
    void SendHalt() {
        zmq_send_int32(send_socket_.get(), kCmdHalt);
        recv_thread_.join();
        // close sockets
        recv_socket_->close();
        recv_socket_.reset(nullptr);
        send_socket_->close();
        send_socket_.reset(nullptr);
        mailbox_.reset(nullptr);
    }
   private:
    void Main() {
        while(true) {
            int type = zmq_recv_int32(recv_socket_.get());
            switch (type) {
            case kCmdTask: {
                int dst = zmq_recv_int32(recv_socket_.get());
                int model_id = zmq_recv_int32(recv_socket_.get());
                std::vector<float> msg = ModelTransferStore::Get().Pop(model_id);
                base::BinStream bin;
                bin << msg;
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
   private:
    std::thread recv_thread_;
    zmq::context_t* context_;
    std::unique_ptr<zmq::socket_t> recv_socket_;
    std::unique_ptr<zmq::socket_t> send_socket_;
    std::unique_ptr<husky::LocalMailbox> mailbox_;
    const WorkerInfo& worker_info_;
};

}  // namespace husky
