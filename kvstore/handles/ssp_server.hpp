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
 * Functor class for SSP in server side
 *
 * Caution:
 * Note that the maximun difference of parameter of each worker in the same iter (each time add 1)
 * should be at most staleness*num_workers*2-2.
 * Push determines the iteration, After each iteration, workers can choose to Pull or not.
 * No need to issue Push/Pull/Push/Pull, Push/Push/Push/Pull should be fine
 *
 * At the begining of each epoch, InitForConsistencyControl must be invoked
 * Push may also be blocked.
 * Now, Setting s to 0, it is still not BSP, workers may see newer parameter than BSP:
 * Iter: 0, 0, 0, 0
 * worker 0 push
 * Iter: 1, 0, 0, 0
 * worker 1 pull will see the push result of worker 0
 */
template <typename Val, typename StorageT>
class SSPServer : public ServerBase {
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
                min_clock_ = 0;
                worker_progress_.clear();
                clock_count_.clear();
                blocked_pulls_.clear();
                blocked_pushes_.clear();
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

        if (push) {  // if is push
            if (src >= worker_progress_.size())
                worker_progress_.resize(src + 1);
            int expected_min_clock = worker_progress_[src] - staleness_;
            if (expected_min_clock <= min_clock_) {  // acceptable staleness so process it
                process_push(kv_id, ts, cmd, src, bin, customer);
            } else {  // blocked to expected_min_clock
                if (blocked_pushes_.size() <= expected_min_clock)
                    blocked_pushes_.resize(expected_min_clock+1);
                blocked_pushes_[expected_min_clock].emplace_back(cmd, src, ts, std::move(bin));
            }
        } else {  // if is pull
            if (src >= worker_progress_.size())
                worker_progress_.resize(src + 1);
            int expected_min_clock = worker_progress_[src] - staleness_;
            if (expected_min_clock <= min_clock_) {  // acceptable staleness so reply it
                if (bin.size()) {  // if bin is empty, don't reply
                    KVPairs<Val> res = retrieve<Val, StorageT>(kv_id, server_id_, bin, store_, cmd>with_min_clock_magic_?cmd-with_min_clock_magic_:cmd, is_vector_);
                    if (cmd > with_min_clock_magic_)
                        Response<Val>(kv_id, ts, cmd, push, src, res, customer, min_clock_);
                    else
                        Response<Val>(kv_id, ts, cmd, push, src, res, customer);
                }
            } else {  // block it to expected_min_clock(i.e. worker_progress_[src] - staleness_)
                if (blocked_pulls_.size() <= expected_min_clock)
                    blocked_pulls_.resize(expected_min_clock + 1);
                blocked_pulls_[expected_min_clock].emplace_back(cmd, src, ts, std::move(bin));
            }
        }
    }
    SSPServer() = delete;
    SSPServer(int server_id, int num_workers, StorageT&& store, bool is_vector, int staleness)
        : server_id_(server_id), num_workers_(num_workers), worker_progress_(num_workers), store_(std::move(store)), is_vector_(is_vector), staleness_(staleness) {}

   private:
    /*
     * Function to process_push
     */
    void process_push(int kv_id, int ts, int cmd, int src, husky::base::BinStream& bin, ServerCustomer* customer) {
        if (bin.size()) {  // if bin is empty, don't reply
            update<Val, StorageT>(kv_id, server_id_, bin, store_, cmd, is_vector_, false);
            Response<Val>(kv_id, ts, cmd, true, src, KVPairs<Val>(), customer);
        }
        if (src >= worker_progress_.size())
            worker_progress_.resize(src + 1);
        int progress = worker_progress_[src];
        if (progress >= clock_count_.size())
            clock_count_.resize(progress + 1);
        clock_count_[progress] += 1;  // add clock_count_
        if (progress == min_clock_ && clock_count_[min_clock_] == num_workers_) {
            min_clock_ += 1;
            // release all push blocked at min_clock_
            if (blocked_pushes_.size() <= min_clock_)
                blocked_pushes_.resize(min_clock_ + 1);
            for (auto& push_pair : blocked_pushes_[min_clock_]) {
                if (std::get<3>(push_pair).size()) {
                    int push_cmd = std::get<0>(push_pair);
                    update<Val, StorageT>(kv_id, server_id_, std::get<3>(push_pair), store_, push_cmd, is_vector_, false);
                    Response<Val>(kv_id, std::get<2>(push_pair), push_cmd, true, std::get<1>(push_pair), KVPairs<Val>(), customer);
                }
            }
            std::vector<std::tuple<int, int, int, husky::base::BinStream>>().swap(blocked_pushes_[min_clock_]);
            // release all pull blocked at min_clock_
            if (blocked_pulls_.size() <= min_clock_)
                blocked_pulls_.resize(min_clock_ + 1);
            for (auto& pull_pair : blocked_pulls_[min_clock_]) {
                if (std::get<3>(pull_pair).size()) {  // if bin is empty, don't reply
                    int pull_cmd = std::get<0>(pull_pair);
                    KVPairs<Val> res = retrieve<Val, StorageT>(kv_id, server_id_, std::get<3>(pull_pair), store_, pull_cmd>with_min_clock_magic_?pull_cmd-with_min_clock_magic_:pull_cmd, is_vector_);
                    if (cmd > with_min_clock_magic_)  // PullChunksWithMinClock
                        Response<Val>(kv_id, std::get<2>(pull_pair), std::get<0>(pull_pair), false, std::get<1>(pull_pair), res, customer, min_clock_);
                    else
                        Response<Val>(kv_id, std::get<2>(pull_pair), std::get<0>(pull_pair), false, std::get<1>(pull_pair), res, customer);
                }
            }
            std::vector<std::tuple<int, int, int, husky::base::BinStream>>().swap(blocked_pulls_[min_clock_]);
        }
        worker_progress_[src] += 1;
    }

   private:
    int num_workers_;
    int staleness_ = 0;

    int min_clock_ = 0;
    std::vector<int> clock_count_;  // TODO: may use round array to reduce the space
    std::vector<int> worker_progress_;
    std::vector<std::vector<std::tuple<int, int, int, husky::base::BinStream>>> blocked_pushes_;   // cmd, src, ts, bin
    std::vector<std::vector<std::tuple<int, int, int, husky::base::BinStream>>> blocked_pulls_;   // cmd, src, ts, bin
    // default storage method is unordered_map
    bool is_vector_ = false;
    // The real storeage
    StorageT store_;
    int server_id_;

    // For init
    int init_count_ = 0;
};

}  // namespace kvstore
