#pragma once

#include <unordered_map>
#include <vector>

#include "husky/base/serialization.hpp"
#include "kvstore/kvmanager.hpp"

#include <iostream>

namespace kvstore {


template<typename Val>
struct KVServerBSPHandle {
    void update(husky::base::BinStream& bin) {
        while (bin.size() > 0) {
            int k;
            Val v;
            bin >> k >> v;
            store_[k] += v;
        }
    }
    KVPairs<Val> retrieve(husky::base::BinStream& bin) {
        KVPairs<Val> res;
        while (bin.size() > 0) {
            int k;
            bin >> k;
            res.keys.push_back(k);
            res.vals.push_back(store_[k]);
        }
        return res;
    }

    // the callback function
    void operator()(int kv_id, int ts, husky::base::BinStream& bin, ServerCustomer* customer, KVServer<Val>* server) {
        bool push;  // push or not
        bin >> push;
        if (push) {  // if is push
            push_count_ += 1;
            if (reply_phase) {  // if is now replying, should update later
                push_buffer_.push_back(std::move(bin));
            } else {  // otherwise, directly update
                int src;
                bin >> src;
                update(bin);
                server->Response(kv_id, ts, push, src, KVPairs<Val>(), customer);
            }
            // if all the push are collected, reply for the pull
            if (push_count_ == num_workers_) {
                push_count_ = 0;
                reply_phase = true;
                for (auto& bin : pull_buffer_) {  // process the pull_buffer_
                    int src;
                    bin >> src;
                    KVPairs<Val> res = retrieve(bin);
                    server->Response(kv_id, ts+1, 0, src, res, customer);
                }
                pull_buffer_.clear();
            }
        } else {  // if is pull
            pull_count_ += 1;
            if (reply_phase) {  // if is now replying, directly reply
                int src;
                bin >> src;
                KVPairs<Val> res = retrieve(bin);
                server->Response(kv_id, ts, push, src, res, customer);
            } else {  // otherwise, reply later
                pull_buffer_.push_back(std::move(bin));
            }
            // if all the pull are replied, change to non reply-phase
            if (pull_count_ == num_workers_) {
                pull_count_ = 0;
                reply_phase = false;
                for (auto& bin : push_buffer_) {  // process the push_buffer_
                    int src;
                    bin >> src;
                    update(bin);
                    server->Response(kv_id, ts+1, 1, src, KVPairs<Val>(), customer);
                }
                push_buffer_.clear();
            }
        }
    }


    KVServerBSPHandle() = delete;
    KVServerBSPHandle(int num_workers): num_workers_(num_workers) {}
private:
    int num_workers_;
    int push_count_ = 0;
    int pull_count_ = 0;

    std::vector<husky::base::BinStream> push_buffer_;
    std::vector<husky::base::BinStream> pull_buffer_;

    bool reply_phase = false;

    // The real storeage
    std::unordered_map<int, Val> store_;
};

}  // namespace kvstore
