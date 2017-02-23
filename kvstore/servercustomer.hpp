#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include "base/serialization.hpp"
#include "core/mailbox.hpp"

namespace kvstore {

/*
 * ServerCustomer is only for KVManager!!!
 */
class ServerCustomer {
   public:
    /*
     * the handle for a received message
     */
    using RecvHandle = std::function<void(int, int, husky::base::BinStream&)>;

    ServerCustomer(husky::LocalMailbox& mailbox, const RecvHandle& recv_handle, int channel_id)
        : mailbox_(mailbox), recv_handle_(recv_handle), channel_id_(channel_id) {}
    ~ServerCustomer() { recv_thread_->join(); }
    void Start();
    void Stop();
    void send(int dst, husky::base::BinStream& bin);

   private:
    void Receiving();

    // mailbox
    husky::LocalMailbox& mailbox_;  // reference to mailbox

    // receive thread and receive handle
    RecvHandle recv_handle_;
    std::unique_ptr<std::thread> recv_thread_;

    // some info
    int channel_id_;
};

}  // namespace kvstore
