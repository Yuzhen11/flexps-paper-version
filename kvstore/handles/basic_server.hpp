#pragma once

#include <unordered_map>

#include "husky/base/serialization.hpp"
#include "kvstore/kvmanager.hpp"

namespace kvstore {

/*
 * The default functor for add operation
 */
template<typename Val>
struct KVServerDefaultAddHandle {
    void operator()(int kv_id, int ts, husky::base::BinStream& bin, ServerCustomer* customer, KVServer<Val>* server) {
        bool push;  // push or not
        int src;
        bin >> push >> src;
        KVPairs<Val> res;
        if (push == true) {  // if is push
            while (bin.size() > 0) {
                int k;
                Val v;
                bin >> k >> v;
                // husky::base::log_msg("[Debug][KVServer] Adding k:"+std::to_string(k)+" v:"+std::to_string(v));
                store_[k] += v;
            }
        } else {  // if is pull
            while (bin.size() > 0) {
                int k;
                bin >> k;
                // husky::base::log_msg("[Debug][KVServer] Getting k:"+std::to_string(k)+"
                // v:"+std::to_string(store_[k]));
                res.keys.push_back(k);
                res.vals.push_back(store_[k]);
            }
        }
        server->Response(kv_id, ts, push, src, res, customer);
    }
    // The real storeage
    std::unordered_map<int, Val> store_;
};

/*
 * The default functor for assign operation
 */
template<typename Val>
struct KVServerDefaultAssignHandle {
    void operator()(int kv_id, int ts, husky::base::BinStream& bin, ServerCustomer* customer, KVServer<Val>* server) {
        bool push;  // push or not
        int src;
        bin >> push >> src;
        KVPairs<Val> res;
        if (push == true) {  // if is push
            while (bin.size() > 0) {
                int k;
                Val v;
                bin >> k >> v;
                // husky::base::log_msg("[Debug][KVServer] Assigning k:"+std::to_string(k)+" v:"+std::to_string(v));
                store_[k] = v;
            }
        } else {  // if is pull
            while (bin.size() > 0) {
                int k;
                bin >> k;
                // husky::base::log_msg("[Debug][KVServer] Getting k:"+std::to_string(k)+"
                // v:"+std::to_string(store_[k]));
                res.keys.push_back(k);
                res.vals.push_back(store_[k]);
            }
        }
        server->Response(kv_id, ts, push, src, res, customer);
    }
    // The real storeage
    std::unordered_map<int, Val> store_;
};

}  // namespace kvstore
