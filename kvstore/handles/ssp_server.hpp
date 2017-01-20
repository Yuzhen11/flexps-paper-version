#pragma once

#include <unordered_map>
#include <vector>
#include <tuple>

#include "husky/base/serialization.hpp"
#include "kvstore/kvmanager.hpp"
#include "kvstore/handles/basic.hpp"

namespace kvstore {

/*
 * Functor class for SSP in server side
 *
 * Caution:
 * 1. Note that the maximun difference of parameter of each worker in the same iter (each time add 1)
 * should be at most staleness*num_workers*2-2.
 * 2. Setting to s = 0, it's like BSP, but the first Pull may not see the same data, unless each worker issue a Push first?
 * 3. Push determines the iteration, After each iteration, workers can choose to Pull or not. 
 * No need to issue Push/Pull/Push/Pull, Push/Push/Push/Pull should be fine
 */
template<typename Val>
struct KVServerSSPHandle {
    // the callback function
    void operator()(int kv_id, int ts, husky::base::BinStream& bin, ServerCustomer* customer, KVServer<Val>* server) {
        bool push;  // push or not
        bin >> push;
        if (push) {  // if is push
            int src;
            bin >> src;
            update(bin, store_);
            server->Response(kv_id, ts, push, src, KVPairs<Val>(), customer);

            if (src >= worker_progress_.size()) worker_progress_.resize(src+1);
            int progress = worker_progress_[src];
            if (progress >= clock_count_.size()) clock_count_.resize(progress+1);
            clock_count_[progress] += 1;  // add clock_count_
            if (progress == min_clock_ && clock_count_[min_clock_] == num_workers_) {
                min_clock_ += 1;
                // release all pull blocked at min_clock_
                if (blocked_pulls_.size() <= min_clock_)
                    blocked_pulls_.resize(min_clock_+1);
                for (auto& pull_pair : blocked_pulls_[min_clock_]) {
                    KVPairs<Val> res = retrieve(std::get<2>(pull_pair), store_);
                    server->Response(kv_id, std::get<1>(pull_pair), 0, std::get<0>(pull_pair), res, customer);
                }
                std::vector<std::tuple<int, int, husky::base::BinStream>>().swap(blocked_pulls_[min_clock_]);
            }
            worker_progress_[src] += 1;
        } else {  // if is pull
            int src;
            bin >> src;
            if (src >= worker_progress_.size()) worker_progress_.resize(src+1);
            int expected_min_lock = worker_progress_[src] - staleness_;
            if (expected_min_lock <= min_clock_) {  // acceptable staleness so reply it
                KVPairs<Val> res = retrieve(bin, store_);
                server->Response(kv_id, ts, push, src, res, customer);
            } else {  // block it to expected_min_lock(i.e. worker_progress_[src] - staleness_)
                if (blocked_pulls_.size() <= expected_min_lock)
                    blocked_pulls_.resize(expected_min_lock+1);
                blocked_pulls_[expected_min_lock].emplace_back(src, ts, std::move(bin));
            }
        }
    }
    KVServerSSPHandle() = delete;
    KVServerSSPHandle(int num_workers, int staleness): num_workers_(num_workers), worker_progress_(num_workers), staleness_(staleness) {}
   private:
    int num_workers_;
    std::vector<int> worker_progress_;
    int min_clock_ = 0;
    std::vector<int> clock_count_;  // TODO: may use round array to reduce the space
    int staleness_ = 0;
    std::vector<std::vector<std::tuple<int, int, husky::base::BinStream>>> blocked_pulls_;
    // The real storeage
    std::unordered_map<int, Val> store_;
};

}  // namespace kvstore
