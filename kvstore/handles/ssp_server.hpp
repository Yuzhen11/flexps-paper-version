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
 * 1. Note that the maximun difference of parameter of each worker in the same iter (each time add 1)
 * should be at most staleness*num_workers*2-2.
 * 2. Setting to s = 0, it's like BSP, but the first Pull may not see the same data, unless each worker issue a Push
 * first?
 * 3. Push determines the iteration, After each iteration, workers can choose to Pull or not.
 * No need to issue Push/Pull/Push/Pull, Push/Push/Push/Pull should be fine
 */
template <typename Val, typename StorageT>
class SSPServer : public ServerBase {
   public:
    // the callback function
    virtual void Process(int kv_id, int ts, husky::base::BinStream& bin, ServerCustomer* customer) override {
        int cmd;
        bool push;  // push or not
        bin >> cmd;
        bin >> push;
        if (push) {  // if is push
            int src;
            bin >> src;
            if (bin.size()) {  // if bin is empty, don't reply
                update<Val, StorageT>(kv_id, server_id_, bin, store_, cmd, is_vector_, is_assign_);
                Response<Val>(kv_id, ts, cmd, push, src, KVPairs<Val>(), customer);
            }
            if (src >= worker_progress_.size())
                worker_progress_.resize(src + 1);
            int progress = worker_progress_[src];
            if (progress >= clock_count_.size())
                clock_count_.resize(progress + 1);
            clock_count_[progress] += 1;  // add clock_count_
            if (progress == min_clock_ && clock_count_[min_clock_] == num_workers_) {
                min_clock_ += 1;
                // release all pull blocked at min_clock_
                if (blocked_pulls_.size() <= min_clock_)
                    blocked_pulls_.resize(min_clock_ + 1);
                for (auto& pull_pair : blocked_pulls_[min_clock_]) {
                    if (std::get<3>(pull_pair).size()) {  // if bin is empty, don't reply
                        KVPairs<Val> res = retrieve<Val, StorageT>(kv_id, server_id_, std::get<3>(pull_pair), store_, std::get<0>(pull_pair), is_vector_);
                        Response<Val>(kv_id, std::get<2>(pull_pair), std::get<0>(pull_pair), 0, std::get<1>(pull_pair), res, customer);
                    }
                }
                std::vector<std::tuple<int, int, int, husky::base::BinStream>>().swap(blocked_pulls_[min_clock_]);
            }
            worker_progress_[src] += 1;
        } else {  // if is pull
            int src;
            bin >> src;
            if (src >= worker_progress_.size())
                worker_progress_.resize(src + 1);
            int expected_min_lock = worker_progress_[src] - staleness_;
            if (expected_min_lock <= min_clock_) {  // acceptable staleness so reply it
                if (bin.size()) {  // if bin is empty, don't reply
                    KVPairs<Val> res = retrieve<Val, StorageT>(kv_id, server_id_, bin, store_, cmd);
                    Response<Val>(kv_id, ts, cmd, push, src, res, customer);
                }
            } else {  // block it to expected_min_lock(i.e. worker_progress_[src] - staleness_)
                if (blocked_pulls_.size() <= expected_min_lock)
                    blocked_pulls_.resize(expected_min_lock + 1);
                blocked_pulls_[expected_min_lock].emplace_back(cmd, src, ts, std::move(bin));
            }
        }
    }
    SSPServer() = delete;
    SSPServer(int server_id, int num_workers, StorageT&& store, bool is_vector, bool is_assign, int staleness)
        : server_id_(server_id), num_workers_(num_workers), worker_progress_(num_workers), store_(std::move(store)), is_vector_(is_vector), is_assign_(is_assign), staleness_(staleness) {}

   private:
    int num_workers_;
    std::vector<int> worker_progress_;
    int min_clock_ = 0;
    std::vector<int> clock_count_;  // TODO: may use round array to reduce the space
    int staleness_ = 0;
    std::vector<std::vector<std::tuple<int, int, int, husky::base::BinStream>>> blocked_pulls_;   // cmd, src, ts, bin
    // default storage method is unordered_map
    bool is_vector_ = false;
    // default update method is assign
    bool is_assign_ = false;
    // The real storeage
    StorageT store_;
    int server_id_;
};

}  // namespace kvstore
