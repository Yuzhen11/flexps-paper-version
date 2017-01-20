#pragma once

#include <unordered_map>

#include "husky/base/serialization.hpp"
#include "kvstore/kvmanager.hpp"
#include "kvstore/ps_lite/sarray.h"
#include "kvstore/handles/basic.hpp"

#include "core/color.hpp"

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
        if (push == true) {  // if is push
            update(bin, store_);
            server->Response(kv_id, ts, push, src, KVPairs<Val>(), customer);
        } else {  // if is pull
            KVPairs<Val> res = retrieve(bin, store_);
            server->Response(kv_id, ts, push, src, res, customer);
        }
    }
    // The real storeage
    std::unordered_map<husky::constants::Key, Val> store_;
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
        if (push == true) {  // if is push
            assign(bin, store_);
            server->Response(kv_id, ts, push, src, KVPairs<Val>(), customer);
        } else {  // if is pull
            KVPairs<Val> res = retrieve(bin, store_);
            server->Response(kv_id, ts, push, src, res, customer);
        }
    }
    // The real storeage
    std::unordered_map<husky::constants::Key, Val> store_;
};

}  // namespace kvstore
