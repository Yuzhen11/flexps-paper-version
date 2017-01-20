#pragma once

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <vector>
#include <limits>

#include "kvpairs.hpp"
#include "workercustomer.hpp"

#include "core/info.hpp"
#include "husky/base/serialization.hpp"
#include "husky/core/mailbox.hpp"

#include "core/color.hpp"

namespace kvstore {

struct RecvKVPairsBase {};

template <typename Val>
struct RecvKVPairs : public RecvKVPairsBase {
    std::vector<KVPairs<Val>> recv_kvs;
};

class KVWorker {
   public:
    using Callback = std::function<void()>;
    template<typename Val>
    using SlicedKVs = std::vector<std::pair<bool, KVPairs<Val>>>;

    KVWorker(const PSInfo& info, husky::LocalMailbox& mailbox)
        : customer_(new WorkerCustomer(mailbox, [this](int kv_id, int ts, husky::base::BinStream& bin,
                                                       bool runCallback) { Process(kv_id, ts, bin, runCallback); },
                                       info.channel_id)),
          info_(info) {
        customer_->Start();
    }
    ~KVWorker() { customer_->Stop(); }

    /*
     * Pushes a list of kv pairs to all server nodes
     */
    template <typename Val>
    int Push(int kv_id,
            const std::vector<husky::constants::Key>& keys,
            const std::vector<Val>& vals,
            const Callback& cb = nullptr) {
        return ZPush(
            kv_id, pslite::SArray<husky::constants::Key>(keys), pslite::SArray<Val>(vals), cb);
    }

    /*
     * zero-copy push
     */
    template <typename Val>
    int ZPush(int kv_id,
             const pslite::SArray<husky::constants::Key>& keys,
             const pslite::SArray<Val>& vals,
             const Callback& cb = nullptr) {
        int ts = customer_->NewRequest(kv_id, info_.num_ps_servers);
        KVPairs<Val> kvs;
        kvs.keys = keys;
        kvs.vals = vals;
        Send(kv_id, ts, true, kvs);
        return ts;
    }

    /*
     * Pulls the values associated with the keys from the server nodes
     */
    template<typename Val>
    int Pull(int kv_id, 
           const std::vector<husky::constants::Key>& keys,
           std::vector<Val>* vals,
           const Callback& cb = nullptr) {
        return Pull_<Val>(kv_id, pslite::SArray<husky::constants::Key>(keys), vals, cb);
    }
    /*
     * zero-copy pull
     */
    template<typename Val>
    int ZPull(int kv_id,
              const pslite::SArray<husky::constants::Key>& keys,
              pslite::SArray<Val>* vals,
              const Callback& cb = nullptr) {
        return Pull_<Val>(kv_id, keys, vals, cb);
    }


    /*
     * \brief Waits until a push or pull has been finished
     */
    void Wait(int kv_id, int timestamp) { customer_->WaitRequest(kv_id, timestamp); }

    /*
     * \brief Engine use this funciton to add process func
     */
    template <typename Val>
    void AddProcessFunc(int kv_id) {
        assert(process_map.find(kv_id) == process_map.end());
        process_map.insert(
            std::make_pair(kv_id, [this](int kv_id, int ts, husky::base::BinStream& bin, bool runCallback) {
                UniqueProcess<Val>(kv_id, ts, bin, runCallback);
            }));  // push the function template in
    }

    void SetMaxKey(int kv_id, husky::constants::Key max_key) {
        GetServerKeyRanges(kv_id, max_key);
    }
   private:
    /*
     * \brief UniqueProcess for every individual kvstore
     */
    template <typename Val>
    void UniqueProcess(int kv_id, int ts, husky::base::BinStream& bin, bool runCallback) {
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
            if (recv_kvs_.find({kv_id, ts}) == recv_kvs_.end()) {
                recv_kvs_[{kv_id, ts}] = new RecvKVPairs<Val>();
            }
            static_cast<RecvKVPairs<Val>*>(recv_kvs_[{kv_id, ts}])->recv_kvs.push_back(kvs);
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
        if (!cb)
            return;
        std::lock_guard<std::mutex> lk(mu_);
        callbacks_[{kv_id, ts}] = cb;
    }
    /*
     * Run the callback
     */
    void RunCallback(int kv_id, int ts) {
        mu_.lock();
        auto it = callbacks_.find({kv_id, ts});
        if (it != callbacks_.end()) {
            mu_.unlock();

            it->second();

            mu_.lock();
            callbacks_.erase(it);
        }
        mu_.unlock();
    }

    template<typename Val, typename C>
    int Pull_(int kv_id, const pslite::SArray<husky::constants::Key>& keys, C* vals, const Callback& cb) {
        auto num_servers = info_.num_ps_servers;
        int ts = customer_->NewRequest(kv_id, num_servers);
        AddCallback(kv_id, ts, [this, kv_id, ts, keys, vals, cb, num_servers]() mutable {
            mu_.lock();
            auto& kvs = static_cast<RecvKVPairs<Val>*>(recv_kvs_[{kv_id, ts}])->recv_kvs;
            mu_.unlock();

            // do check
            size_t total_key = 0, total_val = 0;
            for (const auto& s : kvs) {
              pslite::Range range = pslite::FindRange(keys, s.keys.front(), s.keys.back()+1);
              total_key += s.keys.size();
              total_val += s.vals.size();
            }

            // fill vals and lens
            std::sort(kvs.begin(), kvs.end(), [](
                const KVPairs<Val>& a, const KVPairs<Val>& b) {
                        return a.keys.front() < b.keys.front();
              });
            if (vals->empty()) {
              vals->resize(total_val);
            } else {
            }
            Val* p_vals = vals->data();
            for (const auto& s : kvs) {
              memcpy(p_vals, s.vals.data(), s.vals.size() * sizeof(Val));
              p_vals += s.vals.size();
            }

            mu_.lock();
            delete recv_kvs_[{kv_id, ts}];
            recv_kvs_.erase({kv_id, ts});
            mu_.unlock();
            if (cb) cb();
        });
        KVPairs<Val> kvs; kvs.keys = keys;
        Send(kv_id, ts, false, kvs);
        return ts;
    }

    /*
     * The Send function to send out Push/Pull request
     */
    template<typename Val>
    void Send(int kv_id, int ts, bool push, const KVPairs<Val>& kvs) {
        // slice the message
        SlicedKVs<Val> sliced;
        Slice(kvs, GetServerKeyRanges(kv_id), &sliced);

        for (size_t i = 0; i < sliced.size(); ++ i) {
            husky::base::BinStream bin;
            bool isRequest = true;
            int src = info_.global_id;
            bin << isRequest << kv_id << ts << push << src;
            auto& kvs = sliced[i].second;
            bin << kvs.keys << kvs.vals;
            // husky::LOG_I << CLAY("sending to "+std::to_string(i)+" size: "+std::to_string(bin.size()));
            customer_->send(info_.get_tid(i), bin);
        }
    }


    /*
     * The Slice function to slice the parameters
     */
    template<typename Val>
    void Slice(
        const KVPairs<Val>& send, const std::vector<pslite::Range>& ranges,
        SlicedKVs<Val>* sliced) {
        sliced->resize(ranges.size());
  
        // find the positions in msg.key
        size_t n = ranges.size();
        std::vector<size_t> pos(n+1);
        const husky::constants::Key* begin = send.keys.begin();
        const husky::constants::Key* end = send.keys.end();
        for (size_t i = 0; i < n; ++i) {
          if (i == 0) {
            pos[0] = std::lower_bound(begin, end, ranges[0].begin()) - begin;
            begin += pos[0];
          } else {
          }
          size_t len = std::lower_bound(begin, end, ranges[i].end()) - begin;
          begin += len;
          pos[i+1] = pos[i] + len;
  
          // don't send it to severs for empty kv
          sliced->at(i).first = (len != 0);
        }
        if (send.keys.empty()) return;
  
        // the length of value
        size_t k = 0, val_begin = 0, val_end = 0;
        k = send.vals.size() / send.keys.size();
  
        // slice
        for (size_t i = 0; i < n; ++i) {
          if (pos[i+1] == pos[i]) {
            sliced->at(i).first = false;
            continue;
          }
          sliced->at(i).first = true;
          auto& kv = sliced->at(i).second;
          kv.keys = send.keys.segment(pos[i], pos[i+1]);
          kv.vals = send.vals.segment(pos[i]*k, pos[i+1]*k);
        }
    }
    const std::vector<pslite::Range>& GetServerKeyRanges(int kv_id, 
                                                         husky::constants::Key max_key = std::numeric_limits<husky::constants::Key>::max()) {
        if (kv_id >= server_key_ranges_.size())
            server_key_ranges_.resize(kv_id+1);
      if (server_key_ranges_[kv_id].empty()) {
        auto num_servers_ = info_.num_ps_servers;
        for (int i = 0; i < num_servers_; ++i) {
          server_key_ranges_[kv_id].push_back(pslite::Range(
              max_key / num_servers_ * i,
              max_key / num_servers_ * (i+1)));
        }
      }
      return server_key_ranges_[kv_id];
    }

   private:
    std::vector<std::vector<pslite::Range>> server_key_ranges_;

    // storage for the kvs
    std::unordered_map<std::pair<int, int>, RecvKVPairsBase*> recv_kvs_;  // { <kv_id,ts>, recv_kvs_ }
    // callbacks
    std::unordered_map<std::pair<int, int>, Callback> callbacks_;  // { <kv_id,ts>, callback_ }
    // process function map
    std::unordered_map<int, std::function<void(int, int, husky::base::BinStream&, bool)>>
        process_map;  // {kv_id, process()}
    std::mutex mu_;

    // customer
    std::unique_ptr<WorkerCustomer> customer_;
    PSInfo info_;
};

}  // namespace kvstore
