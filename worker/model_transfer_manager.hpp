#pragma once

#include <thread>

#include "husky/core/worker_info.hpp"
#include "husky/core/zmq_helpers.hpp"
#include "husky/core/mailbox.hpp"
#include "husky/base/exception.hpp"
#include "worker/model_transfer_store.hpp"
#include "core/color.hpp"

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
            zmq::context_t* context);
    ~ModelTransferManager();
    ModelTransferManager(const ModelTransferManager&) = delete;
    ModelTransferManager& operator=(const ModelTransferManager&) = delete;
    void Serve(MailboxEventLoop* const el);
    /*
     * Worker uses SendTask to send the task to ModelTransferManager's event-loop
     */
    void SendTask(int dst, int task_id);
    /*
     * Worker uses SendHalt to halt the ModelTransferManager's event-loop
     */
    void SendHalt();

   private:
    void Main();
   private:
    std::thread recv_thread_;
    zmq::context_t* context_;
    std::unique_ptr<zmq::socket_t> recv_socket_;
    std::unique_ptr<zmq::socket_t> send_socket_;
    std::unique_ptr<husky::LocalMailbox> mailbox_;
    const WorkerInfo& worker_info_;
};

}  // namespace husky
