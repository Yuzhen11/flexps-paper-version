#pragma once

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <vector>

#include "core/info.hpp"
#include "customer.hpp"
#include "husky/base/serialization.hpp"
#include "husky/core/mailbox.hpp"

namespace ml {
namespace ps {

struct PSInfo {
    int channel_id;
    int global_id;
    int num_global_threads;
    int num_ps_servers;
    std::unordered_map<int, int> cluster_id_to_global_id;  // {cluster_id, global_id}

    int get_tid(int cluster_id) { return cluster_id_to_global_id[cluster_id]; }
};

PSInfo info2psinfo(const husky::Info& info) {
    PSInfo psinfo;
    psinfo.channel_id = info.get_task()->get_id();
    psinfo.global_id = info.get_global_id();
    psinfo.num_global_threads = info.get_num_workers();
    psinfo.num_ps_servers = static_cast<husky::PSTask*>(info.get_task())->get_num_ps_servers();
    psinfo.cluster_id_to_global_id = info.get_cluster_global();
    return psinfo;
}

/*
 * Use std::vector first, may replaced by SArray
 */
template <typename Val>
struct KVPairs {
    std::vector<int> keys;
    std::vector<Val> vals;
};

class SimpleApp {
   public:
    void ShutDown() {
        // To shut down, send an empty BinStream to every worker
        for (int i = 0; i < info_.num_global_threads; ++i) {
            int dst = info_.get_tid(i);
            husky::base::BinStream bin;
            this->obj_->send(dst, bin);
        }
    }

   protected:
    using RecvHandle = std::function<void(int ts, husky::base::BinStream& bin)>;
    SimpleApp(const PSInfo& info, husky::LocalMailbox& mailbox, const RecvHandle& recv_handle)
        : info_(info), obj_(new Customer(mailbox, recv_handle, info.num_global_threads, info.channel_id)) {}
    std::unique_ptr<Customer> obj_;
    PSInfo info_;

   private:
};

/*
 * A worker node that can Push/Pull key-value pairs to/from server nodes
 */
template <typename Val>
class KVWorker : public SimpleApp {
   public:
    using Callback = std::function<void()>;
    using SimpleApp::obj_;  // to avoid to many this->
    KVWorker(const PSInfo& info, husky::LocalMailbox& mailbox)
        : SimpleApp(info, mailbox, [this](int ts, husky::base::BinStream& bin) { Process(ts, bin); }) {
        obj_->Start();
    }
    /*
     * Pushes a list of kv pairs to all server nodes
     *
     * it's a non-blocking call.
     */
    int Push(const std::vector<int>& keys, const std::vector<Val>& vals, const Callback& cb = nullptr) {
        auto num_servers = info_.num_ps_servers;
        int ts = obj_->NewRequest(num_servers);
        AddCallback(ts, cb);
        // create the send buffer
        // 1. serialize k/v into the send_buffer
        // 2. send the message
        husky::base::BinStream bins[num_servers];
        bool isRequest = true;
        bool isPush = true;
        int src = info_.global_id;
        for (int i = 0; i < num_servers; ++i) {
            // isRequest, ts, isPush, src
            bins[i] << isRequest << ts << isPush << src;
        }
        for (int i = 0; i < keys.size(); ++i) {
            int dst = keys[i] % num_servers;  // use the basic hash partition
            // k, v
            bins[dst] << keys[i] << vals[i];  // serialize
        }
        for (int i = 0; i < num_servers; ++i) {
            this->obj_->send(info_.get_tid(i), bins[i]);  // send
        }
        return ts;
    }

    /*
     * Pulls the values associated with the keys from the server nodes
     *
     * it's a non-blocking call, use wait to block on that
     */
    int Pull(const std::vector<int>& keys, std::vector<Val>* vals, const Callback& cb = nullptr) {
        auto num_servers = info_.num_ps_servers;
        int ts = obj_->NewRequest(num_servers);
        AddCallback(ts, cb);

        husky::base::BinStream bins[num_servers];
        bool isRequest = true;
        bool isPush = false;
        int src = info_.global_id;
        for (int i = 0; i < num_servers; ++i) {
            bins[i] << isRequest << ts << isPush << src;
        }
        for (int i = 0; i < keys.size(); ++i) {
            int dst = keys[i] % num_servers;
            bins[dst] << keys[i];
        }
        for (int i = 0; i < num_servers; ++i) {
            this->obj_->send(info_.get_tid(i), bins[i]);  // send
        }
        // add callback
        // TODO, since here we don't use sarray in ps-lite, deep copy of vector of keys are needed
        AddCallback(ts, [this, ts, keys, vals, cb]() mutable {
            mu_.lock();
            auto& kvs = recv_kvs_[ts];
            mu_.unlock();

            // TODO may incur a lot copy
            std::vector<std::pair<int, Val>> v;
            for (const auto& s : kvs) {
                for (int i = 0; i < s.keys.size(); ++i) {
                    v.push_back({s.keys[i], s.vals[i]});
                }
            }
            std::sort(v.begin(), v.end(),
                      [](const std::pair<int, Val>& p1, const std::pair<int, Val>& p2) { return p1.first < p2.first; });
            vals->resize(keys.size());
            for (int i = 0; i < keys.size(); ++i) {
                int k = keys[i];
                auto p = std::find_if(v.begin(), v.end(), [k](const std::pair<int, Val>& p) { return p.first == k; });
                assert(p != v.end());
                (*vals)[i] = p->second;
            }
            mu_.lock();
            recv_kvs_.erase(ts);
            mu_.unlock();
            if (cb)
                cb();
        });
        return ts;
    }
    /*
     * Waits until a push or pull has been finished
     */
    void Wait(int timestamp) { obj_->WaitRequest(timestamp); }

    /*
     * The callback function registered to customer
     */
    void Process(int ts, husky::base::BinStream& bin) {
        bool push;  // push or not
        int src;
        bin >> push;
        bin >> src;
        if (push == true)
            ;                      // if is push
        else if (push == false) {  // if is pull
            KVPairs<Val> kvs;
            // Format: keys, values
            bin >> kvs.keys >> kvs.vals;
            mu_.lock();
            recv_kvs_[ts].push_back(kvs);
            mu_.unlock();
        }
        // If all the servers response, run the callback
        if (this->obj_->NumResponse(ts) == info_.num_ps_servers - 1) {
            RunCallback(ts);
        }
    }

    /*
     * Add a callback for a request
     */
    void AddCallback(int ts, const Callback& cb) {
        if (!cb)
            return;
        std::lock_guard<std::mutex> lk(mu_);
        callbacks_[ts] = cb;
    }
    /*
     * Run the callback
     */
    void RunCallback(int ts) {
        mu_.lock();
        auto it = callbacks_.find(ts);
        if (it != callbacks_.end()) {
            mu_.unlock();

            it->second();

            mu_.lock();
            callbacks_.erase(it);
        }
        mu_.unlock();
    }

   private:
    // storage for the kvs
    std::unordered_map<int, std::vector<KVPairs<Val>>> recv_kvs_;
    // callbacks
    std::unordered_map<int, Callback> callbacks_;
    std::mutex mu_;
};

// forward declaration
template <typename Val>
struct KVServerDefaultHandle;

/*
 * A server node for maintaining kv pairs
 */
template <typename Val>
class KVServer : public SimpleApp {
   public:
    KVServer(const PSInfo& info, husky::LocalMailbox& mailbox)
        : SimpleApp(info, mailbox, [this](int ts, husky::base::BinStream& bin) { Process(ts, bin); }),
          request_handle_(KVServerDefaultHandle<Val>()) {
        // Start the recv thread once everything is set up
        this->obj_->Start();
    }
    ~KVServer() = default;

    /*
     * the handle to process a push/pull request from a worker
     */
    using ReqHandle = std::function<void(int, husky::base::BinStream&, KVServer<Val>*)>;

    /*
     * response to the push/pull request
     * The whole callback process is:
     * process -> request_handle_ -> Response
     */
    void Response(int ts, bool push, int src, const KVPairs<Val>& res) {
        husky::base::BinStream bin;
        bool isRequest = false;
        // isRequest, ts, isPush, src
        bin << isRequest << ts << push << src;

        bin << res.keys << res.vals;
        this->obj_->send(src, bin);
    }

   private:
    // internal receive handle
    void Process(int ts, husky::base::BinStream& bin) { request_handle_(ts, bin, this); }
    // request handle
    ReqHandle request_handle_;
};

/*
 * The default KVServer handle
 * An example handle adding pushed kv into store
 */
template <typename Val>
struct KVServerDefaultHandle {
    void operator()(int ts, husky::base::BinStream& bin, KVServer<Val>* server) {
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
                store[k] += v;
            }
        } else {  // if is pull
            while (bin.size() > 0) {
                int k;
                bin >> k;
                // husky::base::log_msg("[Debug][KVServer] Getting k:"+std::to_string(k)+"
                // v:"+std::to_string(store[k]));
                res.keys.push_back(k);
                res.vals.push_back(store[k]);
            }
        }
        server->Response(ts, push, src, res);
    }

    // The actual storage
    std::unordered_map<int, Val> store;
};

}  // namespace ps
}  // namespace ml
