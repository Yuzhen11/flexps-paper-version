#pragma once

#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>
#include "base/serialization.hpp"
#include "core/common/mailbox.hpp"

namespace ml {
namespace ps {

/*
 * Use the same name with ps-lite
 *
 * It has its own receiving thread to poll messages from LocalMailbox
 * and invoke the callback
 * 
 */
class Customer {
public:
    using RecvHandle = std::function<void(int ts, husky::base::BinStream& bin)>;
    // TODO What is the minimun struct the a Customer object needs
    Customer(husky::LocalMailbox& mailbox, const RecvHandle& recv_handle)
        :mailbox_(mailbox), recv_handle_(recv_handle) {
        // spawn a new thread to recevive
        recv_thread_ = std::unique_ptr<std::thread>(new std::thread(&Customer::Receiving, this));
    }
    ~Customer() {
        recv_thread_->join();
    }

    int NewRequest(int num_responses) {
        std::lock_guard<std::mutex> lk(tracker_mu_);
        tracker_.push_back({num_responses, 0});
        return tracker_.size()-1;
    }
    void WaitRequest(int timestamp) {
        std::unique_lock<std::mutex> lk(tracker_mu_);
        tracker_cond_.wait(lk, [this, timestamp] {
            return tracker_[timestamp].first == tracker_[timestamp].second;
        });
    }
    int NumResponse(int timestamp) {
        std::lock_guard<std::mutex> lk(tracker_mu_);
        return tracker_[timestamp].second;
    }
    void send(int dst, husky::base::BinStream& bin) {
        mailbox_.send(dst, channel_id, 0, bin);
    }
private:
    void Receiving() {
        // poll and recv from mailbox
        while (mailbox_.poll(channel_id, 0)) {
            auto bin = mailbox_.recv(channel_id, 0);
            // empty message means exit
            if (bin.size() == 0)
                break;
            // Format: isRequest, ts, push, src, k, v...
            // response: 0, ts, push, src, keys, vals ; handled by worker
            // request: 1, ts, push, src, k, v, k, v... ; handled by server
            bool isRequest;
            int ts;
            bin >> isRequest >> ts;
            // invoke the callback
            recv_handle_(ts, bin);
            if (isRequest == false) {
                std::lock_guard<std::mutex> lk(tracker_mu_);
                tracker_[ts].second += 1;
                tracker_cond_.notify_all();
            }
        }
    }

    husky::LocalMailbox& mailbox_;  // reference to mailbox

    RecvHandle recv_handle_;
    std::unique_ptr<std::thread> recv_thread_;

    // tracker
    std::mutex tracker_mu_;
    std::condition_variable tracker_cond_;
    std::vector<std::pair<int,int>> tracker_;

    // Debug
    static const int channel_id = 0;

};

}  // namespace ps
}  // namespace ml
