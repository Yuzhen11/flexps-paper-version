#pragma once

#include <thread>
#include <vector>
#include <unordered_map>

#include "kvpairs.hpp"

#include "core/mailbox.hpp"
#include "base/serialization.hpp"

namespace kvstore {

/*
 * ServerCustomer is only for KVStoreManager!!!
 */
class ServerCustomer {
public:
    /*
     * the handle for a received message
     */
    using RecvHandle = std::function<void(int,int,husky::base::BinStream&)>;

    ServerCustomer(husky::LocalMailbox& mailbox, const RecvHandle& recv_handle, int channel_id)
        : mailbox_(mailbox),
          recv_handle_(recv_handle),
          channel_id_(channel_id){
    }
    ~ServerCustomer() {
        recv_thread_->join();
    }
    void Start() {
        // spawn a new thread to recevive
        recv_thread_ = std::unique_ptr<std::thread>(new std::thread(&ServerCustomer::Receiving, this));
    }
    void Stop() {
        husky::base::BinStream bin;  // send an empty BinStream
        mailbox_.send(mailbox_.get_thread_id(), channel_id_, 0, bin);
    }
    void send(int dst, husky::base::BinStream& bin) {
        mailbox_.send(dst, channel_id_, 0, bin);
    }
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

/*
 * KVServerBase: Base class for different kvserver
 */
class KVServerBase {
public:
    virtual void HandleAndReply(int, int, husky::base::BinStream&, ServerCustomer* customer) = 0;
};
template<typename Val>
class KVStoreServer : public KVServerBase {
public:
    KVStoreServer() = default;
    ~KVStoreServer() = default;

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

    /*
     * Handle the BinStream and then reply
     */
    virtual void HandleAndReply(int kv_id, int ts, husky::base::BinStream& bin, ServerCustomer* customer) override {
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
                store[k] = v;
            }
        } else {  // if is pull
            while (bin.size() > 0) {
                int k;
                bin >> k;
                // husky::base::log_msg("[Debug][KVServer] Getting k:"+std::to_string(k)+" v:"+std::to_string(store[k]));
                res.keys.push_back(k);
                res.vals.push_back(store[k]);
            }
        }
        Response(kv_id, ts, push, src, res, customer);
    }
private:
    // The real storeage
    std::unordered_map<int, Val> store;
};

/*
 * KVStoreManager manages many KVStoreServer, so different types of data can be stored
 */
class KVStoreManager {
public:
    KVStoreManager(husky::LocalMailbox& mailbox, int channel_id)
        : customer_(new ServerCustomer(mailbox, [this](int kv_id, int ts, husky::base::BinStream& bin){ Process(kv_id, ts, bin); }, channel_id)) {
        customer_->Start();
    }
    ~KVStoreManager() {
        // stop the customer
        customer_->Stop();
        // kv_store_ will be automatically deleted
    }

    /*
     * create different kv_store
     * make sure all the kvstore is set up before the actuall workload
     */
    template<typename Val>
    int create_kvstore() {
        kv_store_.insert(std::make_pair(kv_id_, std::unique_ptr<KVStoreServer<Val>>(new KVStoreServer<Val>())));
        return kv_id_++;
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
    int kv_id_ = 0;
};

}  // namespace kvstore
