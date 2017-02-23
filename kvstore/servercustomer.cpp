#include "kvstore/servercustomer.hpp"

namespace kvstore {

void ServerCustomer::Start() {
    // spawn a new thread to recevive
    recv_thread_ = std::unique_ptr<std::thread>(new std::thread(&ServerCustomer::Receiving, this));
}

void ServerCustomer::Stop() {
    husky::base::BinStream bin;  // send an empty BinStream
    mailbox_.send(mailbox_.get_thread_id(), channel_id_, 0, bin);
}

void ServerCustomer::send(int dst, husky::base::BinStream& bin) { mailbox_.send(dst, channel_id_, 0, bin); }

void ServerCustomer::Receiving() {
    // poll and recv from mailbox
    int num_finished_workers = 0;
    while (mailbox_.poll(channel_id_, 0)) {
        auto bin = mailbox_.recv(channel_id_, 0);
        if (bin.size() == 0) {
            break;
        }
        // Format: isRequest, kv_id, ts, push, src, k, v...
        // response: 0, kv_id, ts, push, src, keys, vals ; handled by worker
        // request: 1, kv_id, ts, push, src, k, v, k, v... ; handled by server
        bool isRequest;
        int kv_id;
        int ts;
        bin >> isRequest >> kv_id >> ts;
        // invoke the callback
        recv_handle_(kv_id, ts, bin);
    }
}

}  // namespace kvstore
