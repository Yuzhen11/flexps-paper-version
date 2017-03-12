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
    void Response(int kv_id, int ts, int cmd, bool push, int src, const KVPairs<Val>& res, ServerCustomer* customer) {
        husky::base::BinStream bin;
        bool isRequest = false;
        // isRequest, kv_id, ts, isPush, src
        bin << isRequest << kv_id << ts << cmd << push << src;
        if (cmd == 2 && push == false) {  // enable zero-copy for local Pull
            KVPairs<Val>* p = new KVPairs<Val>();  // delete by worker
            p->keys = res.keys;
            p->vals = res.vals;
            bin << reinterpret_cast<std::uintptr_t>(p);
        } else {
            bin << res.keys << res.vals;
        }
        customer->send(src, bin);
    }
};

/*
 * The default functor for assign operation
 */
template <typename Val, typename StorageT>
class DefaultUpdateServer : public ServerBase {
   public:
    DefaultUpdateServer() = delete;
    DefaultUpdateServer(int kv_id, int server_id, StorageT&& store, bool is_vector, bool is_assign) : server_id_(server_id), kv_id_(kv_id), store_(std::move(store)), is_vector_(is_vector), is_assign_(is_assign) {}

    virtual void Process(int kv_id, int ts, husky::base::BinStream& bin, ServerCustomer* customer) override {
        int cmd;
        bool push;  // push or not
        int src;
        bin >> cmd >> push >> src;
        assert(cmd != 4);  // no InitForConsistencyControl
        if (push == true) {  // if is push
            if (bin.size()) {  // if bin is empty, don't reply
                update<Val, StorageT>(kv_id, server_id_, bin, store_, cmd, is_vector_, is_assign_);
                
                Response<Val>(kv_id, ts, cmd, push, src, KVPairs<Val>(), customer);
            }
        } else {  // if is pull
            if (bin.size()) {  // if bin is empty, don't reply

                KVPairs<Val> res;
                res = retrieve<Val, StorageT>(kv_id, server_id_, bin, store_, cmd, is_vector_); 
                
                Response<Val>(kv_id, ts, cmd, push, src, res, customer);
            }
        }
    }
   private:
    int kv_id_;
    int server_id_;
    // The real storeage
    StorageT store_;
    // default storage method is map, so store_method_vector = false;
    bool is_vector_ = false;
    // default update method is add
    bool is_assign_ = true;
};

}  // namespace kvstore
