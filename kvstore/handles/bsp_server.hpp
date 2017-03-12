#pragma once

#include <tuple>
#include <unordered_map>
#include <vector>

#include "husky/base/serialization.hpp"
#include "kvstore/handles/basic.hpp"
#include "kvstore/kvmanager.hpp"

#include "kvstore/handles/basic_server.hpp"

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
 *
 * 2017.3.12 Fix a major bug: Cannot use push_count_/pull_count_ only to calculate
 * how many workers have done the push/pull, since next push/pull will be counted.
 * So, push_progress_/pull_progress_ are created to track the progress of push/pull of each workers.
 *
 * At the begining of each epoch, InitForConsistencyControl must be invoked
 */
template <typename Val, typename StorageT>
class BSPServer : public ServerBase {
   public:
    // the callback function
    virtual void Process(int kv_id, int ts, husky::base::BinStream& bin, ServerCustomer* customer) override {
        int cmd;
        bool push;  // push or not
        int src;
        bin >> cmd;
        bin >> push;
        bin >> src;
        if (cmd == 4) {  // InitForConsistencyControl
            if (init_count_ == 0) {  // reset the buffer when the first init message comes
                push_iter_ = 0;
                pull_iter_ = 0;
                push_count_ = 0;
                pull_count_ = 0;
                push_progress_.clear();
                pull_progress_.clear();
                blocked_pulls_.clear();
                blocked_pushes_.clear();
                reply_phase_ = init_reply_phase_;
            }
            init_count_ += 1;
            if (init_count_ == num_workers_) {
                init_count_ = 0;
            }
            Response<Val>(kv_id, ts, cmd, push, src, KVPairs<Val>(), customer);  // Reply directly
            return;
        }

        // husky::LOG_I << CLAY("src: " + std::to_string(src) + 
        //     " reply_phase: " + std::to_string(reply_phase_) +
        //     " push: " + std::to_string(push) + 
        //     " push_count: " + std::to_string(push_count_) + 
        //     " pull_count: " + std::to_string(pull_count_));
        if (push) {  // if is push
            if (reply_phase_) {  // if is replying
                blocked_pushes_.emplace_back(cmd, src, ts, std::move(bin));
            } else {
                if (src >= push_progress_.size())
                    push_progress_.resize(src + 1);
                if (push_iter_ == push_progress_[src]) {  // first src
                    // update
                    if (bin.size()) {  // if bin is empty, don't reply
                        update<Val, StorageT>(kv_id, server_id_, bin, store_, cmd, is_vector_, false);
                        Response<Val>(kv_id, ts, cmd, push, src, KVPairs<Val>(), customer);
                    }
                    push_progress_[src] += 1;
                    push_count_ += 1;
                    if (push_count_ == num_workers_) {
                        // release the blocked pull
                        for (auto& pull_pair : blocked_pulls_) {
                            if (std::get<3>(pull_pair).size()) {
                                KVPairs<Val> res = retrieve<Val, StorageT>(kv_id, server_id_, std::get<3>(pull_pair), store_, std::get<0>(pull_pair), is_vector_);
                                Response<Val>(kv_id, std::get<2>(pull_pair), std::get<0>(pull_pair), 0, std::get<1>(pull_pair), res, customer);
                            }
                            int pull_src = std::get<1>(pull_pair);
                            if (pull_src >= pull_progress_.size())
                                pull_progress_.resize(pull_src+1);
                            pull_progress_[pull_src] += 1;
                        }
                        pull_count_ = blocked_pulls_.size();
                        blocked_pulls_.clear();
                        reply_phase_ = true;
                        push_iter_ += 1;
                    }
                } else {
                    blocked_pushes_.emplace_back(cmd, src, ts, std::move(bin));
                }
            }
        } else {
            if (reply_phase_) {
                if (src >= pull_progress_.size())
                    pull_progress_.resize(src + 1);
                if (pull_iter_ == pull_progress_[src]) {  // first src
                    // pull
                    if (bin.size()) {  // if bin is empty, don't reply
                        KVPairs<Val> res = retrieve<Val, StorageT>(kv_id, server_id_, bin, store_, cmd);
                        Response<Val>(kv_id, ts, cmd, push, src, res, customer);
                    }
                    pull_progress_[src] += 1;
                    pull_count_ += 1;
                    if (pull_count_ == num_workers_) {
                        // release the blocked push
                        for (auto& push_pair : blocked_pushes_) {
                            if (std::get<3>(push_pair).size()) {
                                update<Val, StorageT>(kv_id, server_id_, std::get<3>(push_pair), store_, std::get<0>(push_pair), is_vector_, false);
                                Response<Val>(kv_id, std::get<2>(push_pair), std::get<0>(push_pair), 1, std::get<1>(push_pair), KVPairs<Val>(), customer);
                            }
                            int push_src = std::get<1>(push_pair);
                            if (push_src >= push_progress_.size())
                                push_progress_.resize(push_src+1);
                            push_progress_[push_src] += 1;
                        }
                        push_count_ = blocked_pushes_.size();
                        blocked_pushes_.clear();
                        reply_phase_ = false;
                        pull_iter_ += 1;
                    }
                } else {
                    blocked_pulls_.emplace_back(cmd, src, ts, std::move(bin));
                }
            } else {
                blocked_pulls_.emplace_back(cmd, src, ts, std::move(bin));
            }
        }
    }

    BSPServer() = delete;
    BSPServer(int server_id, int num_workers, StorageT&& store, bool is_vector) : 
        server_id_(server_id), num_workers_(num_workers), store_(std::move(store)), is_vector_(is_vector) {}
    BSPServer(int server_id, int num_workers, StorageT&& store, bool is_vector, bool init_reply_phase) : 
        server_id_(server_id), num_workers_(num_workers), store_(std::move(store)), is_vector_(is_vector), init_reply_phase_(init_reply_phase) {}

   private:
    int num_workers_;

    int push_iter_ = 0;
    int push_count_ = 0;
    std::vector<int> push_progress_;
    std::vector<std::tuple<int, int, int, husky::base::BinStream>> blocked_pushes_;   // cmd, src, ts, bin
    int pull_iter_ = 0;
    int pull_count_ = 0;
    std::vector<int> pull_progress_;
    std::vector<std::tuple<int, int, int, husky::base::BinStream>> blocked_pulls_;   // cmd, src, ts, bin

    bool init_reply_phase_ = true;
    bool reply_phase_ = true;
    // default storage method is unordered_map
    bool is_vector_ = false;
    // The real storeage
    StorageT store_;
    int server_id_;

    // For init
    int init_count_ = 0;
};

}  // namespace kvstore
