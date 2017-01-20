#pragma once

#include <unordered_map>
#include <vector>

#include "husky/base/serialization.hpp"
#include "kvstore/kvmanager.hpp"
#include "kvstore/handles/basic.hpp"

namespace kvstore {

/*
 * Functor class for BSP in server side
 *
 * In mode reply_phase == 0:
 * Process push requests, Buffer pull requests when push are not complete.
 * When push complete, switch to reply_phase, reply the buffered pull. 
 *
 * In mode reply_phase == 1:
 * Process pull requests, Buffer push requests when pull are not complete.
 * When pull complete, switch to non reply_phase, process the buffered push.
 * 
 * Note:
 * User code should like:
 * Pull, Push, Pull, Push ...
 *
 * If user code starts with Push, set the initial reply_phase to false.
 * If user code starts with Pull, set the initial reply_phase to true.
 *
 * Need to wait for the Push request at least before the next Pull.
 * Otherwise fast worker may issue two consecutive Pull.
 */
template<typename Val>
struct KVServerBSPHandle {
    // the callback function
    void operator()(int kv_id, int ts, husky::base::BinStream& bin, ServerCustomer* customer, KVServer<Val>* server) {
        bool push;  // push or not
        bin >> push;
        if (push) {  // if is push
            push_count_ += 1;
            if (reply_phase_) {  // if is now replying, should update later
                push_buffer_.push_back(std::move(bin));
            } else {  // otherwise, directly update
                int src;
                bin >> src;
                update(bin, store_);
                server->Response(kv_id, ts, push, src, KVPairs<Val>(), customer);
            }
            // if all the push are collected, reply for the pull
            if (push_count_ == num_workers_) {
                push_count_ = 0;
                reply_phase_ = true;
                for (auto& bin : pull_buffer_) {  // process the pull_buffer_
                    int src;
                    bin >> src;
                    KVPairs<Val> res = retrieve(bin, store_);
                    server->Response(kv_id, ts+1, 0, src, res, customer);
                }
                pull_buffer_.clear();
            }
        } else {  // if is pull
            pull_count_ += 1;
            if (reply_phase_) {  // if is now replying, directly reply
                int src;
                bin >> src;
                KVPairs<Val> res = retrieve(bin, store_);
                server->Response(kv_id, ts, push, src, res, customer);
            } else {  // otherwise, reply later
                pull_buffer_.push_back(std::move(bin));
            }
            // if all the pull are replied, change to non reply-phase
            if (pull_count_ == num_workers_) {
                pull_count_ = 0;
                reply_phase_ = false;
                for (auto& bin : push_buffer_) {  // process the push_buffer_
                    int src;
                    bin >> src;
                    update(bin, store_);
                    server->Response(kv_id, ts+1, 1, src, KVPairs<Val>(), customer);
                }
                push_buffer_.clear();
            }
        }
    }


    KVServerBSPHandle() = delete;
    KVServerBSPHandle(int num_workers): num_workers_(num_workers) {}
    KVServerBSPHandle(int num_workers, bool reply_phase): num_workers_(num_workers), reply_phase_(reply_phase) {}
private:
    int num_workers_;
    int push_count_ = 0;
    int pull_count_ = 0;

    std::vector<husky::base::BinStream> push_buffer_;
    std::vector<husky::base::BinStream> pull_buffer_;

    bool reply_phase_ = true;

    // The real storeage
    std::unordered_map<int, Val> store_;
};

}  // namespace kvstore
