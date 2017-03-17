#include "kvstore/workercustomer.hpp"

namespace kvstore {

void WorkerCustomer::Start() {
    // spawn a new thread to recevive
    recv_thread_ = std::unique_ptr<std::thread>(new std::thread(&WorkerCustomer::Receiving, this));
}
void WorkerCustomer::Stop() {
    husky::base::BinStream bin;  // send an empty BinStream
    mailbox_.send(mailbox_.get_thread_id(), channel_id_, 0, bin);
    recv_thread_->join();
}

int WorkerCustomer::NewRequest(int kv_id, int num_responses) {
    std::lock_guard<std::mutex> lk(tracker_mu_);
    if (kv_id >= tracker_.size())
        tracker_.resize(kv_id + 1);
    tracker_[kv_id].push_back({num_responses, 0});
    return tracker_[kv_id].size() - 1;
}
void WorkerCustomer::WaitRequest(int kv_id, int timestamp) {
    std::unique_lock<std::mutex> lk(tracker_mu_);
    tracker_cond_.wait(lk, [this, kv_id, timestamp] {
        return tracker_[kv_id][timestamp].first == tracker_[kv_id][timestamp].second;
    });
}
int WorkerCustomer::NumResponse(int kv_id, int timestamp) {
    std::lock_guard<std::mutex> lk(tracker_mu_);
    return tracker_[kv_id][timestamp].second;
}
void WorkerCustomer::send(int dst, husky::base::BinStream& bin) { mailbox_.send(dst, channel_id_, 0, bin); }

void WorkerCustomer::Receiving() {
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
        tracker_mu_.lock();
        bool runCallback = tracker_[kv_id][ts].second == tracker_[kv_id][ts].first - 1 ? true : false;
        tracker_mu_.unlock();
        // invoke the callback
        recv_handle_(kv_id, ts, bin, runCallback);
        {
            std::lock_guard<std::mutex> lk(tracker_mu_);
            tracker_[kv_id][ts].second += 1;
            if (tracker_[kv_id][ts].second == tracker_[kv_id][ts].first)
                tracker_cond_.notify_all();
        }
    }
}

}  // namespace kvstore
