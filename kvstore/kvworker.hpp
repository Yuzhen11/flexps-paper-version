#pragma once

#include <cassert>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "workercustomer.hpp"
#include "kvpairs.hpp"

#include "core/info.hpp"
#include "core/mailbox.hpp"
#include "base/serialization.hpp"

namespace kvstore {

struct RecvKVPairsBase {
};

template<typename Val>
struct RecvKVPairs : public RecvKVPairsBase {
    std::vector<KVPairs<Val>> recv_kvs;
};

class KVWorker {
public:
    using Callback = std::function<void()>;
    KVWorker(const PSInfo& info, husky::LocalMailbox& mailbox)
        : customer_(new WorkerCustomer(mailbox, [this](int kv_id, int ts, husky::base::BinStream& bin, bool runCallback){ Process(kv_id, ts, bin, runCallback); }, info.channel_id)),
          info_(info) {
        customer_->Start();
    }
    ~KVWorker() {
        customer_->Stop();
    }

    /*
     * Pushes a list of kv pairs to all server nodes
     * 
     * it's a non-blocking call.
     * a template function
     */
    template<typename Val>
    int Push(int kv_id, const std::vector<int>& keys, const std::vector<Val>& vals, const Callback& cb = nullptr) {
        auto num_servers = info_.num_ps_servers;
        int ts = customer_->NewRequest(kv_id, num_servers);
        AddCallback(kv_id, ts, cb);
        // create the send buffer
        // 1. serialize k/v into the send_buffer
        // 2. send the message
        husky::base::BinStream bins[num_servers];
        bool isRequest = true;
        bool isPush = true;
        int src = info_.global_id;
        for (int i = 0; i < num_servers; ++ i) {
            // isRequest, kv_id, ts, isPush, src
            bins[i] << isRequest << kv_id << ts << isPush << src;
        }
        for (int i = 0; i < keys.size(); ++ i) {
            int dst = keys[i]%num_servers;  // use the basic hash partition
            // k, v
            bins[dst] << keys[i] << vals[i];  // serialize
        }
        for (int i = 0; i < num_servers; ++ i) {
            // husky::base::log_msg("[Debug][Push]: sending to: "+std::to_string(info_.get_tid(i)));
            customer_->send(info_.get_tid(i), bins[i]);  // send
        }
        return ts;
    }

    template<typename Val>
    int PushLocal(int kv_id, int dst, const std::vector<int>& keys, const std::vector<Val>& vals, const Callback& cb = nullptr) {
        int ts = customer_->NewRequest(kv_id, 1);
        AddCallback(kv_id, ts, cb);
        // create the send buffer
        // 1. serialize k/v into the send_buffer
        // 2. send the message
        husky::base::BinStream bin;
        bool isRequest = true;
        bool isPush = true;
        int src = info_.global_id;
        // isRequest, kv_id, ts, isPush, src
        bin << isRequest << kv_id << ts << isPush << src;
        for (int i = 0; i < keys.size(); ++ i) {
            bin << keys[i] << vals[i];
        }
        customer_->send(info_.get_tid(dst), bin); // send
        return ts;
    }

    /*
     * Pulls the values associated with the keys from the server nodes
     *
     * it's a non-blocking call, use wait to block on that
     * a template function
     */
    template<typename Val>
    int Pull(int kv_id, const std::vector<int>& keys, std::vector<Val>* vals, const Callback& cb = nullptr) {
        auto num_servers = info_.num_ps_servers;
        int ts = customer_->NewRequest(kv_id, num_servers);
        AddCallback(kv_id, ts, cb);
        
        husky::base::BinStream bins[num_servers];
        bool isRequest = true;
        bool isPush = false;
        int src = info_.global_id;
        for (int i = 0; i < num_servers; ++ i) {
            bins[i] << isRequest << kv_id << ts << isPush << src;
        }
        for (int i = 0; i < keys.size(); ++ i) {
            int dst = keys[i]%num_servers;
            bins[dst] << keys[i];
        }
        for (int i = 0; i < num_servers; ++ i) {
            // husky::base::log_msg("[Debug][Pull]: sending to: "+std::to_string(info_.get_tid(i)));
            customer_->send(info_.get_tid(i), bins[i]);  // send
        }
        // add callback
        // TODO, since here we don't use sarray in ps-lite, deep copy of vector of keys are needed
        AddCallback(kv_id, ts, [this, kv_id, ts, keys, vals, cb, num_servers]() mutable {
            // check whether I usually need to handle it
            // if not, return
            // husky::base::log_msg("NumResponse: "+std::to_string(customer_->NumResponse(kv_id, ts)));
            // if (customer_->NumResponse(kv_id, ts) < num_servers - 1)
            //     return;

            mu_.lock();
            auto& kvs = static_cast<RecvKVPairs<Val>*>(recv_kvs_[{kv_id,ts}])->recv_kvs;
            mu_.unlock();
            
            // TODO may incur a lot copy
            std::vector<std::pair<int, Val>> v;
            for (const auto& s : kvs) {
                for (int i = 0; i < s.keys.size(); ++ i) {
                    v.push_back({s.keys[i], s.vals[i]});
                }
            }
            std::sort(v.begin(), v.end(), [](const std::pair<int, Val>& p1, const std::pair<int, Val>& p2) {
                return p1.first < p2.first;
            });
            vals->resize(keys.size());
            for (int i = 0; i < keys.size(); ++ i) {
                int k = keys[i];
                auto p = std::find_if(v.begin(), v.end(), [k](const std::pair<int, Val>& p){ return p.first == k; });
                assert(p!=v.end());
                (*vals)[i] = p->second;
            }
            mu_.lock();
            delete recv_kvs_[{kv_id, ts}];
            recv_kvs_.erase({kv_id,ts});
            mu_.unlock();
            if (cb) cb();
        });
        return ts;
    }

    template<typename Val>
    int PullLocal(int kv_id, int dst, const std::vector<int>& keys, std::vector<Val>* vals, const Callback& cb = nullptr) {
        int ts = customer_->NewRequest(kv_id, 1);
        AddCallback(kv_id, ts, cb);
        
        husky::base::BinStream bin;
        bool isRequest = true;
        bool isPush = false;
        int src = info_.global_id;
        bin << isRequest << kv_id << ts << isPush << src;
        for (int i = 0; i < keys.size(); ++ i) {
            bin << keys[i];
        }
        customer_->send(info_.get_tid(dst), bin);  // send
        // add callback
        // TODO, since here we don't use sarray in ps-lite, deep copy of vector of keys are needed
        AddCallback(kv_id, ts, [this, kv_id, ts, keys, vals, cb]() mutable {
            mu_.lock();
            auto& kvs = static_cast<RecvKVPairs<Val>*>(recv_kvs_[{kv_id,ts}])->recv_kvs;
            mu_.unlock();
            
            // TODO may incur a lot copy
            std::vector<std::pair<int, Val>> v;
            for (const auto& s : kvs) {
                for (int i = 0; i < s.keys.size(); ++ i) {
                    v.push_back({s.keys[i], s.vals[i]});
                }
            }
            std::sort(v.begin(), v.end(), [](const std::pair<int, Val>& p1, const std::pair<int, Val>& p2) {
                return p1.first < p2.first;
            });
            vals->resize(keys.size());
            for (int i = 0; i < keys.size(); ++ i) {
                int k = keys[i];
                auto p = std::find_if(v.begin(), v.end(), [k](const std::pair<int, Val>& p){ return p.first == k; });
                assert(p!=v.end());
                (*vals)[i] = p->second;
            }
            mu_.lock();
            delete recv_kvs_[{kv_id, ts}];
            recv_kvs_.erase({kv_id,ts});
            mu_.unlock();
            if (cb) cb();
        });
        return ts;
    }
    /*
     * \brief Waits until a push or pull has been finished
     */
    void Wait(int kv_id, int timestamp) { customer_->WaitRequest(kv_id, timestamp); }

    /*
     * \brief Engine use this funciton to add process func
     */
    template<typename Val>
    void AddProcessFunc(int kv_id) {
        assert(process_map.find(kv_id) == process_map.end());
        process_map.insert(std::make_pair(kv_id, [this](int kv_id, int ts, husky::base::BinStream& bin, bool runCallback){UniqueProcess<Val>(kv_id, ts, bin, runCallback);}));  // push the function template in
    }
private:
    /*
     * \brief UniqueProcess for every individual kvstore
     */
    template<typename Val>
    void UniqueProcess(int kv_id, int ts, husky::base::BinStream& bin, bool runCallback) {
        bool push;  // push or not
        int src;
        bin >> push;
        bin >> src;
        if (push == true);  // if is push
        else if (push == false) {  // if is pull
            KVPairs<Val> kvs;
            // Format: keys, values
            bin >> kvs.keys >> kvs.vals;
            mu_.lock();
            if (recv_kvs_.find({kv_id, ts}) == recv_kvs_.end()) {
                recv_kvs_[{kv_id,ts}] = new RecvKVPairs<Val>();
            }
            static_cast<RecvKVPairs<Val>*>(recv_kvs_[{kv_id,ts}])->recv_kvs.push_back(kvs);
            mu_.unlock();
        }
        // If all the servers response, run the callback
        // if (customer_->NumResponse(kv_id, ts) == info_.num_ps_servers - 1) {
        //     RunCallback(kv_id, ts);
        // }
        if (runCallback)
            RunCallback(kv_id, ts);
    }
    /*
     * The callback function registered to customer
     * Dispatch the response
     */
    void Process(int kv_id, int ts, husky::base::BinStream& bin, bool runCallback) {
        assert(process_map.find(kv_id) != process_map.end());
        process_map[kv_id](kv_id, ts, bin, runCallback);
    }

    /*
     * Add a callback for a request
     */
    void AddCallback(int kv_id, int ts, const Callback& cb) {
        if (!cb) return;
        std::lock_guard<std::mutex> lk(mu_);
        callbacks_[{kv_id,ts}] = cb;
    }
    /*
     * Run the callback
     */
    void RunCallback(int kv_id, int ts) {
        mu_.lock();
        auto it = callbacks_.find({kv_id,ts});
        if (it != callbacks_.end()) {
            mu_.unlock();

            it->second();

            mu_.lock();
            callbacks_.erase(it);
        }
        mu_.unlock();
    }


    // storage for the kvs
    std::unordered_map<std::pair<int,int>, RecvKVPairsBase*> recv_kvs_;  // { <kv_id,ts>, recv_kvs_ }
    // callbacks
    std::unordered_map<std::pair<int,int>, Callback> callbacks_;  // { <kv_id,ts>, callback_ }
    // process function map
    std::unordered_map<int, std::function<void(int,int,husky::base::BinStream&,bool)>> process_map;  // {kv_id, process()}
    std::mutex mu_;

    // customer
    std::unique_ptr<WorkerCustomer> customer_;
    PSInfo info_;
};


}  // namespace kvstore
