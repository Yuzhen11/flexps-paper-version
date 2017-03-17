#include "kvstore/servercustomer.hpp"

namespace kvstore {

void ServerCustomer::Start() {
    // spawn a new thread to recevive
    recv_thread_ = std::unique_ptr<std::thread>(new std::thread(&ServerCustomer::Receiving, this));
}

void ServerCustomer::Stop() {
    husky::base::BinStream bin;  // send an empty BinStream
    mailbox_.send(mailbox_.get_thread_id(), channel_id_, 0, bin);
    recv_thread_->join();
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
        // Format: kv_id, ts, cmd, push, src, data
        int kv_id;
        int ts;
        bin >> kv_id >> ts;
        // invoke the callback
        recv_handle_(kv_id, ts, bin);
    }
}

}  // namespace kvstore
