#pragma once

#include <unordered_map>
#include <vector>
#include <tuple>

#include "husky/base/serialization.hpp"
#include "kvstore/kvmanager.hpp"

namespace kvstore {

/*
 * Functor class for SSP in server side
 *
 * Note that the maximun difference of parameter of each worker in the same iter (each time add 1)
 * should be at most staleness*num_workers*2-2.
 *
 * Note:
 * User code should like:
 * Pull, Push, Pull, Push ...
 */
template<typename Val>
struct KVServerSSPHandle {
    // update function for push
    void update(husky::base::BinStream& bin) {
        while (bin.size() > 0) {
            int k;
            Val v;
            bin >> k >> v;
            store_[k] += v;
        }
    }
    // retrieve function for pull
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
            int src;
            bin >> src;
            update(bin);
            server->Response(kv_id, ts, push, src, KVPairs<Val>(), customer);

            if (src >= worker_progress_.size()) worker_progress_.resize(src+1);
            int progress = worker_progress_[src];
            if (progress >= clock_count_.size()) clock_count_.resize(progress+1);
            clock_count_[progress] += 1;  // add clock_count_
            if (progress == min_clock_ && clock_count_[min_clock_] == num_workers_) {
                min_clock_ += 1;
                for (auto& pull_pair : blocked_pulls_) {
                    KVPairs<Val> res = retrieve(std::get<2>(pull_pair));
                    server->Response(kv_id, std::get<1>(pull_pair), 0, std::get<0>(pull_pair), res, customer);
                }
                blocked_pulls_.clear();
            }
            worker_progress_[src] += 1;
        } else {  // if is pull
            int src;
            bin >> src;
            if (src >= worker_progress_.size()) worker_progress_.resize(src+1);
            if (worker_progress_[src] - min_clock_ <= staleness_) {  // acceptable staleness so reply it
                KVPairs<Val> res = retrieve(bin);
                server->Response(kv_id, ts, push, src, res, customer);
            } else {  // block it
                blocked_pulls_.emplace_back(src, ts, std::move(bin));
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
    std::vector<std::tuple<int, int, husky::base::BinStream>> blocked_pulls_;
    // The real storeage
    std::unordered_map<int, Val> store_;
};

}  // namespace kvstore
