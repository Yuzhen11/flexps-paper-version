#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include "base/serialization.hpp"
#include "core/mailbox.hpp"

namespace kvstore {

/*
 * Use the same name with ps-lite
 *
 * It has its own receiving thread to poll messages from LocalMailbox
 * and invoke the callback.
 *
 * Users (KVWorker and KVServer) need to give
 * a callback function
 *
 */
class WorkerCustomer {
   public:
    /*
     * the handle for a received message
     */
    using RecvHandle = std::function<void(int, int, husky::base::BinStream&, bool)>;

    WorkerCustomer(husky::LocalMailbox& mailbox, const RecvHandle& recv_handle, int channel_id)
        : mailbox_(mailbox), recv_handle_(recv_handle), channel_id_(channel_id) {}
    ~WorkerCustomer() { recv_thread_->join(); }

    /*
     * Function to Start and Stop the customer
     */
    void Start();
    void Stop();

    int NewRequest(int kv_id, int num_responses);
    void WaitRequest(int kv_id, int timestamp);
    int NumResponse(int kv_id, int timestamp);
    void send(int dst, husky::base::BinStream& bin); 
   private:
    void Receiving();

    // mailbox
    husky::LocalMailbox& mailbox_;  // reference to mailbox

    // receive thread and receive handle
    RecvHandle recv_handle_;
    std::unique_ptr<std::thread> recv_thread_;

    // tracker
    std::mutex tracker_mu_;
    std::condition_variable tracker_cond_;
    std::vector<std::vector<std::pair<int, int>>> tracker_;  // kv_id, ts, <expected, current>

    // some info
    int channel_id_;
    int total_workers_;
};

}  // namespace kvstore
