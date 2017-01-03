#pragma once

#include <thread>
#include <unordered_map>
#include <vector>

#include "kvpairs.hpp"

#include "husky/base/exception.hpp"
#include "husky/base/serialization.hpp"
#include "husky/core/mailbox.hpp"

namespace kvstore {

/*
 * ServerCustomer is only for KVManager!!!
 */
class ServerCustomer {
   public:
    /*
     * the handle for a received message
     */
    using RecvHandle = std::function<void(int, int, husky::base::BinStream&)>;

    ServerCustomer(husky::LocalMailbox& mailbox, const RecvHandle& recv_handle, int channel_id)
        : mailbox_(mailbox), recv_handle_(recv_handle), channel_id_(channel_id) {}
    ~ServerCustomer() { recv_thread_->join(); }
    void Start() {
        // spawn a new thread to recevive
        recv_thread_ = std::unique_ptr<std::thread>(new std::thread(&ServerCustomer::Receiving, this));
    }
    void Stop() {
        husky::base::BinStream bin;  // send an empty BinStream
        mailbox_.send(mailbox_.get_thread_id(), channel_id_, 0, bin);
    }
    void send(int dst, husky::base::BinStream& bin) { mailbox_.send(dst, channel_id_, 0, bin); }

   private:
    void Receiving() {
        // poll and recv from mailbox
        int num_finished_workers = 0;
        while (mailbox_.poll(channel_id_, 0)) {
            auto bin = mailbox_.recv(channel_id_, 0);
            if (bin.size() == 0) {
                break;
            }
            // Format: isRequest, kv_id, ts, push, src, k, v...
            // response: 0, kv_id, ts, push, src, keys, vals ; handled by worker
            // request: 1, kv_id, ts, push, src, k, v, k, v... ; handled by server
            bool isRequest;
            int kv_id;
            int ts;
            bin >> isRequest >> kv_id >> ts;
            // invoke the callback
            recv_handle_(kv_id, ts, bin);
        }
    }

    // mailbox
    husky::LocalMailbox& mailbox_;  // reference to mailbox

    // receive thread and receive handle
    RecvHandle recv_handle_;
    std::unique_ptr<std::thread> recv_thread_;

    // some info
    int channel_id_;
};


// forward declaration
template <typename Val>
class KVServer;
// template alias
template<typename Val>
using ReqHandle = std::function<void(int, int, husky::base::BinStream&, ServerCustomer*, KVServer<Val>*)>;

/*
 * KVServerBase: Base class for different kvserver
 */
class KVServerBase {
   public:
    virtual void HandleAndReply(int, int, husky::base::BinStream&, ServerCustomer* customer) = 0;
};
template <typename Val>
class KVServer : public KVServerBase {
   public:

    KVServer() = delete;
    KVServer(const ReqHandle<Val>& request_handler): request_handler_(request_handler){}
    ~KVServer() = default;

    /*
     * Handle the BinStream and then reply
     */
    virtual void HandleAndReply(int kv_id, int ts, husky::base::BinStream& bin, ServerCustomer* customer) override {
        request_handler_(kv_id, ts, bin, customer, this);
    }

    /*
     * response to the push/pull request
     * The whole callback process is:
     * process -> HandleAndReply -> Response
     */
    void Response(int kv_id, int ts, bool push, int src, const KVPairs<Val>& res, ServerCustomer* customer) {
        husky::base::BinStream bin;
        bool isRequest = false;
        // isRequest, kv_id, ts, isPush, src
        bin << isRequest << kv_id << ts << push << src;

        bin << res.keys << res.vals;
        customer->send(src, bin);
    }

   private:
    ReqHandle<Val> request_handler_;

};

/*
 * KVManager manages many KVServer, so different types of data can be stored
 */
class KVManager {
   public:
    KVManager(husky::LocalMailbox& mailbox, int channel_id)
        : customer_(new ServerCustomer(
              mailbox, [this](int kv_id, int ts, husky::base::BinStream& bin) { Process(kv_id, ts, bin); },
              channel_id)) {
        customer_->Start();
    }
    ~KVManager() {
        // stop the customer
        customer_->Stop();
        // kv_store_ will be automatically deleted
    }

    /*
     * create different kv_store
     * make sure all the kvstore is set up before the actual workload
     */
    template <typename Val>
    void CreateKVManager(int kv_id, const ReqHandle<Val>& request_handler) {
        kv_store_.insert(std::make_pair(kv_id, std::unique_ptr<KVServer<Val>>(new KVServer<Val>(request_handler))));
    }

   private:
    /*
     * Internal receive handle to dispatch the request
     *
     */
    void Process(int kv_id, int ts, husky::base::BinStream& bin) {
        assert(kv_store_.find(kv_id) != kv_store_.end());
        kv_store_[kv_id]->HandleAndReply(kv_id, ts, bin, customer_.get());
    }

    // customer for communication
    std::unique_ptr<ServerCustomer> customer_;
    std::unordered_map<int, std::unique_ptr<KVServerBase>> kv_store_;
};

// The default functor for add operation, used by PSTask
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

// The default functor for assign operation
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
