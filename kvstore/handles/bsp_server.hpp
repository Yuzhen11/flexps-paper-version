#pragma once

#include <iostream>
#include <sstream>
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
 * At the begining of each epoch, InitForConsistencyControl must be invoked
 *
 * Push/Pull from different iterations will be stored to handle the case that
 * worker that doesn't need data from this partition can advance the iterations
 * so there may be more push/pull from this worker
 *
 * Workflow:
 * if push:
 *   if reply phase or push that's too advanced
 *     block
 *   else
 *     handle and response
 *     if receive all push in this iter
 *       release all buffered pull in pull_iter
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
                push_count_.clear();
                pull_count_.clear();
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
        if (cmd >= consistency_control_off_magic_) {  // Without consistency_control
            cmd %= consistency_control_off_magic_;
            if (push == true) {  // if is push
                if (bin.size()) {  // if bin is empty, don't reply
                    update<Val, StorageT>(kv_id, server_id_, bin, store_, cmd, is_vector_, false);
                    Response<Val>(kv_id, ts, cmd, push, src, KVPairs<Val>(), customer);
                }
            } else {  // if is pull
                if (bin.size()) {  // if bin is empty, don't reply
                    KVPairs<Val> res;
                    res = retrieve<Val, StorageT>(kv_id, server_id_, bin, store_, cmd, is_vector_); 
                    Response<Val>(kv_id, ts, cmd, push, src, res, customer);
                }
            }
            return;
        }

        // print_debug(push);

        if (push) {  // if is push
            if (src >= push_progress_.size())
                push_progress_.resize(src + 1);
            int progress = push_progress_[src];
            if (progress >= push_count_.size())
                push_count_.resize(progress + 1);
            push_count_[progress] += 1;
            push_progress_[src] += 1;
            // if in reply phase or the progress is too advanced, buffere it
            if (reply_phase_ || push_iter_ != progress) {  // buffer it
                if (progress >= blocked_pushes_.size())
                    blocked_pushes_.resize(progress + 1);
                blocked_pushes_[progress].emplace_back(cmd, src, ts, std::move(bin));
            } else {  // first src in push_iter_, reply
                // update
                if (bin.size()) {  // if bin is empty, don't reply
                    update<Val, StorageT>(kv_id, server_id_, bin, store_, cmd, is_vector_, false);
                    Response<Val>(kv_id, ts, cmd, push, src, KVPairs<Val>(), customer);
                }
                if (push_count_[push_iter_] == num_workers_) {  // if collect all
                    // release the blocked pull
                    if (blocked_pulls_.size() > pull_iter_) {
                        for (auto& pull_pair : blocked_pulls_[pull_iter_]) {
                            if (std::get<3>(pull_pair).size()) {
                                KVPairs<Val> res = retrieve<Val, StorageT>(kv_id, server_id_, std::get<3>(pull_pair), store_, std::get<0>(pull_pair), is_vector_);
                                Response<Val>(kv_id, std::get<2>(pull_pair), std::get<0>(pull_pair), 0, std::get<1>(pull_pair), res, customer);
                            }
                        }
                        std::vector<std::tuple<int, int, int, husky::base::BinStream>>().swap(blocked_pulls_[pull_iter_]);
                    } else {
                    }
                    reply_phase_ = true;
                    push_iter_ += 1;
                }
            }
        } else {
            if (src >= pull_progress_.size())
                pull_progress_.resize(src + 1);
            int progress = pull_progress_[src];
            if (progress >= pull_count_.size())
                pull_count_.resize(progress + 1);
            pull_count_[progress] += 1;
            pull_progress_[src] += 1;
            if (!reply_phase_ || pull_iter_ != progress) {  // buffer it
                if (progress >= blocked_pulls_.size())
                    blocked_pulls_.resize(progress + 1);
                blocked_pulls_[progress].emplace_back(cmd, src, ts, std::move(bin));
            } else { // first src in pull_iter_, reply
                if (bin.size()) {
                    KVPairs<Val> res = retrieve<Val, StorageT>(kv_id, server_id_, bin, store_, cmd, is_vector_);
                    Response<Val>(kv_id, ts, cmd, push, src, res, customer);
                }
                if (pull_count_[pull_iter_] == num_workers_) {
                    // release the blocked push
                    if (blocked_pushes_.size() > push_iter_) {
                        for (auto& push_pair : blocked_pushes_[push_iter_]) {
                            if (std::get<3>(push_pair).size()) {
                                update<Val, StorageT>(kv_id, server_id_, std::get<3>(push_pair), store_, std::get<0>(push_pair), is_vector_, false);
                                Response<Val>(kv_id, std::get<2>(push_pair), std::get<0>(push_pair), 1, std::get<1>(push_pair), KVPairs<Val>(), customer);
                            }
                        }
                        std::vector<std::tuple<int, int, int, husky::base::BinStream>>().swap(blocked_pushes_[push_iter_]);
                    }
                    reply_phase_ = false;
                    pull_iter_ += 1;
                }
            }
        }
    }

    BSPServer() = delete;
    BSPServer(int server_id, int num_workers, StorageT&& store, bool is_vector) : 
        server_id_(server_id), num_workers_(num_workers), store_(std::move(store)), is_vector_(is_vector) {}
    BSPServer(int server_id, int num_workers, StorageT&& store, bool is_vector, bool init_reply_phase) : 
        server_id_(server_id), num_workers_(num_workers), store_(std::move(store)), is_vector_(is_vector), init_reply_phase_(init_reply_phase) {}

   private:

    void print_debug(bool push) {
        // debug
        std::stringstream ss;
        ss << "reply_phase: " << reply_phase_;
        ss << "\npush_progress: ";
        for (int i = 0; i < push_progress_.size(); ++ i) {
            ss << push_progress_[i] << " ";
        }
        ss << "\npush_count: ";
        for (int i = 0; i < push_count_.size(); ++ i) {
            ss << push_count_[i] << " ";
        }
        // ss << "\nblocked_push size: " << blocked_pushes_.size();
        ss << "\npush_iter: " << push_iter_;
        ss << "\npull_progress: ";
        for (int i = 0; i < pull_progress_.size(); ++ i) {
            ss << pull_progress_[i] << " ";
        }
        ss << "\npull_count: ";
        for (int i = 0; i < pull_count_.size(); ++ i) {
            ss << pull_count_[i] << " ";
        }
        // ss << "\nblocked_pull size: " << blocked_pulls_.size();
        ss << "\npull_iter: " << pull_iter_;
        ss << "\nnext is: " << (push?"push":"pull");
        // husky::LOG_I << CLAY(ss.str());
        std::cout << ss.str();
    };
   private:
    int num_workers_;

    int push_iter_ = 0;
    std::vector<int> push_count_;
    std::vector<int> push_progress_;
    std::vector<std::vector<std::tuple<int, int, int, husky::base::BinStream>>> blocked_pushes_;   // cmd, src, ts, bin
    int pull_iter_ = 0;
    std::vector<int> pull_count_;
    std::vector<int> pull_progress_;
    std::vector<std::vector<std::tuple<int, int, int, husky::base::BinStream>>> blocked_pulls_;   // cmd, src, ts, bin

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
