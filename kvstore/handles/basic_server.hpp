#pragma once

#include <unordered_map>

#include "husky/base/serialization.hpp"
#include "kvstore/handles/basic.hpp"
#include "kvstore/kvmanager.hpp"
#include "kvstore/ps_lite/sarray.h"

#include "core/color.hpp"

#include "kvstore/servercustomer.hpp"

namespace kvstore {

class ServerBase {
   public:
    virtual void Process(int kv_id, int ts, husky::base::BinStream& bin, ServerCustomer* customer) = 0;
    /*
     * response to the push/pull request
     * The whole callback process is:
     * process -> HandleAndReply -> Response
     */
    template<typename Val>
    void Response(int kv_id, int ts, bool push, int src, const KVPairs<Val>& res, ServerCustomer* customer) {
        husky::base::BinStream bin;
        bool isRequest = false;
        // isRequest, kv_id, ts, isPush, src
        bin << isRequest << kv_id << ts << push << src;

        bin << res.keys << res.vals;
        customer->send(src, bin);
    }
};

/*
 * The default functor for add operation
 */
template <typename Val>
class DefaultAddServer : public ServerBase {
   public:
    virtual void Process(int kv_id, int ts, husky::base::BinStream& bin, ServerCustomer* customer) override {
        bool push;  // push or not
        int src;
        bin >> push >> src;
        if (push == true) {  // if is push
            update(bin, store_);
            Response<Val>(kv_id, ts, push, src, KVPairs<Val>(), customer);
        } else {  // if is pull
            KVPairs<Val> res = retrieve(bin, store_);
            Response<Val>(kv_id, ts, push, src, res, customer);
        }
    }
   private:
    // The real storeage
    std::unordered_map<husky::constants::Key, Val> store_;
};

/*
 * The default functor for assign operation
 */
template <typename Val>
class DefaultAssignServer : public ServerBase {
   public:
    virtual void Process(int kv_id, int ts, husky::base::BinStream& bin, ServerCustomer* customer) override {
        bool push;  // push or not
        int src;
        bin >> push >> src;
        if (push == true) {  // if is push
            assign(bin, store_);
            Response<Val>(kv_id, ts, push, src, KVPairs<Val>(), customer);
        } else {  // if is pull
            KVPairs<Val> res = retrieve(bin, store_);
            Response<Val>(kv_id, ts, push, src, res, customer);
        }
    }
   private:
    // The real storeage
    std::unordered_map<husky::constants::Key, Val> store_;
};

}  // namespace kvstore
