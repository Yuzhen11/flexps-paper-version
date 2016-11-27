#pragma once

#include <functional>
#include <mutex>
#include <condition_variable>
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
    using RecvHandle = std::function<void(int,int, husky::base::BinStream&)>;

    WorkerCustomer(husky::LocalMailbox& mailbox, const RecvHandle& recv_handle, int channel_id)
        : mailbox_(mailbox),
          recv_handle_(recv_handle),
          channel_id_(channel_id){
    }
    ~WorkerCustomer() {
        recv_thread_->join();
    }
    void Start() {
        // spawn a new thread to recevive
        recv_thread_ = std::unique_ptr<std::thread>(new std::thread(&WorkerCustomer::Receiving, this));
    }
    void Stop() {
        husky::base::BinStream bin;  // send an empty BinStream
        mailbox_.send(mailbox_.get_thread_id(), channel_id_, 0, bin);
    }

    int NewRequest(int kv_id, int num_responses) {
        std::lock_guard<std::mutex> lk(tracker_mu_);
        if (kv_id >= tracker_.size()) 
            tracker_.resize(kv_id+1);
        tracker_[kv_id].push_back({num_responses, 0});
        return tracker_[kv_id].size()-1;
    }
    void WaitRequest(int kv_id, int timestamp) {
        std::unique_lock<std::mutex> lk(tracker_mu_);
        tracker_cond_.wait(lk, [this, kv_id, timestamp] {
            return tracker_[kv_id][timestamp].first == tracker_[kv_id][timestamp].second;
        });
    }
    int NumResponse(int kv_id, int timestamp) {
        std::lock_guard<std::mutex> lk(tracker_mu_);
        return tracker_[kv_id][timestamp].second;
    }
    void send(int dst, husky::base::BinStream& bin) {
        mailbox_.send(dst, channel_id_, 0, bin);
    }
private:
    void Receiving() {
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
            if (isRequest == false) {
                std::lock_guard<std::mutex> lk(tracker_mu_);
                tracker_[kv_id][ts].second += 1;
                tracker_cond_.notify_all();
            }
        }
    }

    // mailbox
    husky::LocalMailbox& mailbox_;  // reference to mailbox

    // receive thread and receive handle
    RecvHandle recv_handle_;
    std::unique_ptr<std::thread> recv_thread_;

    // tracker
    std::mutex tracker_mu_;
    std::condition_variable tracker_cond_;
    std::vector<std::vector<std::pair<int,int>>> tracker_;  // kv_id, ts, <expected, current>

    // some info
    int channel_id_;
    int total_workers_;

};

}  // namespace kvstore
