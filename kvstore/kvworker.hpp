#pragma once

#include <algorithm>
#include <cassert>
#include <limits>
#include <unordered_map>
#include <vector>

#include "kvpairs.hpp"
#include "workercustomer.hpp"
#include "range_manager.hpp"

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

template <typename Val>
struct RecvKVPairsWithMinClock : public RecvKVPairsBase {
    std::vector<std::pair<KVPairs<Val>, int>> recv_kvs;  // (kvpairs, min_clock)
};

/*
 * cmd:
 * 0: Push/Pull
 * 1: PushChunks/PullChunks
 * 2: local zero-copy Push/Pull
 * 3: local zero-copy PushChunks/PullChunks
 * 4: InitForConsistencyControl
 * 11: PullChunksWithMinClock
 * 13: PullChunksWithMinClock + local zero-copy
 * 100+k: consistency_control off
 *
 * consistency_control_off_magic_:100
 * with_min_clock_magic_: 10
 * local_zero_copy_magic_: 2
 */
class KVWorker {
   public:
    using Callback = std::function<void()>;
    template <typename Val>
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
     * This function is specially used for SSP/BSP consistency control
     * where in the beginning of each epoch, the state in the server side needs to be reset.
     *
     * When using PS ssp, InitForConsistencyControl must be called in each worker threads
     * in the beginning of every epoch.
     */
    int InitForConsistencyControl(int kv_id) {
        int num_servers = RangeManager::Get().GetNumServers();
        int ts = customer_->NewRequest(kv_id, num_servers);
        int src = info_.global_id;
        int cmd = 4;  // special cmd
        bool push = true;
        for (int i = 0; i < num_servers; ++ i) {
            husky::BinStream bin;
            bin << kv_id << ts << cmd << push << src;
            customer_->send(info_.get_tid(i), bin);
        }
        return ts;
    }

    /*
     * Push a list of kv pairs to all server nodes
     *
     * @param kv_id the kv_id users want to handle with
     * @param keys the keys
     * @param vals the vals
     * @param send_all whether need to send to all servers.
     * If it is false, server side control will be disable (ssp, bsp)
     * If it is true, empty messages will be sent to servers which have no partitions, so make sure that these servers won't reply
     * @param local_zero_copy enable local zero copy or not
     * @param consistency_control enable consistency control or not
     * @param cb callback function
     */
    template <typename Val>
    int Push(int kv_id, const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals,
             bool send_all = true, bool local_zero_copy = true, bool consistency_control = true, const Callback& cb = nullptr) {
        return ZPush(kv_id, pslite::SArray<husky::constants::Key>(keys), pslite::SArray<Val>(vals), send_all, local_zero_copy, consistency_control, cb);
    }

    /*
     * zero-copy push
     */
    template <typename Val>
    int ZPush(int kv_id, const pslite::SArray<husky::constants::Key>& keys, const pslite::SArray<Val>& vals,
              bool send_all = true, bool local_zero_copy = true, bool consistency_control = true, const Callback& cb = nullptr) {
        // 1. slice
        KVPairs<Val> kvs;
        kvs.keys = keys;
        kvs.vals = vals;
        SlicedKVs<Val> sliced;
        Slice_(kvs, RangeManager::Get().GetServerKeyRanges(kv_id), &sliced);
        // 2. get ts
        int ts = GetTimestamp_(kv_id, sliced);
        // 3. send
        Send_(kv_id, ts, true, sliced, send_all, local_zero_copy, consistency_control);
        return ts;
    }

    /*
     * Pulls the values associated with the keys from the server nodes
     *
     * @param kv_id the kv_id users want to handle with
     * @param keys the keys
     * @param vals the vals
     * @param send_all whether need to send for all servers, if it is false, server side control will be disable (ssp, bsp)
     * @param local_zero_copy enable local zero copy or not
     * @param consistency_control enable consistency control or not
     * @param cb callback function
     */
    template <typename Val>
    int Pull(int kv_id, const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals,
             bool send_all = true, bool local_zero_copy = true, bool consistency_control = true, const Callback& cb = nullptr) {
        return Pull_<Val>(kv_id, pslite::SArray<husky::constants::Key>(keys), vals, send_all, local_zero_copy, consistency_control,cb);
    }

    /*
     * zero-copy pull
     */
    template <typename Val>
    int ZPull(int kv_id, const pslite::SArray<husky::constants::Key>& keys, pslite::SArray<Val>* vals,
              bool send_all = true, bool local_zero_copy = true, bool consistency_control = true, const Callback& cb = nullptr) {
        return Pull_<Val>(kv_id, keys, vals, send_all, local_zero_copy, consistency_control, cb);
    }

    /*
     * Push a list of chunk to server
     *
     * @param kv_id the kv_id users want to handle with
     * @param chunk_ids The chunk_ids need to be pushed
     * @param chunks The pointer to real chunks
     * @param send_all whether needs to send to all servers
     * @param local_zero_copy enable local zero copy or not
     * @param consistency_control enable consistency control or not
     * @param cb callback function
     */
    template <typename Val>
    int PushChunks(int kv_id, const std::vector<size_t>& chunk_ids, const std::vector<std::vector<Val>*>& chunks,
             bool send_all = true, bool local_zero_copy = true, bool consistency_control = true, const Callback& cb = nullptr) {
        assert(chunk_ids.size() == chunks.size());
        // 1. partition
        std::vector<size_t> pos = PartitionChunks_(kv_id, chunk_ids);
        // 2. get ts
        int ts = GetTimestampChunk_(kv_id, pos, send_all);
        // 3. send
        SendChunks_(kv_id, ts, true, chunk_ids, chunks, pos, send_all, local_zero_copy, false, consistency_control);
        return ts;
    }

    /*
     * Pull a list of chunks from server
     *
     * @param kv_id the kv_id users want to handle with
     * @param chunk_ids The chunk_ids that needs to be pulled
     * @param chunks The chunks provided must be created beforehand
     * @param send_all whether needs to send to all servers
     * @param local_zero_copy enable local zero copy or not
     * @param consistency_control enable consistency control or not
     * @param cb callback function
     */
    template<typename Val>
    int PullChunks(int kv_id, const std::vector<size_t>& chunk_ids, const std::vector<std::vector<Val>*>& chunks,
             bool send_all = true, bool local_zero_copy = true, bool consistency_control = true, const Callback& cb = nullptr) {
        assert(chunk_ids.size() == chunks.size());
        // 1. partition
        std::vector<size_t> pos = PartitionChunks_(kv_id, chunk_ids);
        // 2. get ts
        int ts = GetTimestampChunk_(kv_id, pos, send_all);
        AddCallback(kv_id, ts, [this, kv_id, ts, chunk_ids, chunks, cb]() {
            mu_.lock();
            auto& kvs = static_cast<RecvKVPairs<Val>*>(recv_kvs_[{kv_id, ts}])->recv_kvs;
            mu_.unlock();

            size_t chunk_size = RangeManager::Get().GetChunkSize(kv_id);
            size_t chunk_num = RangeManager::Get().GetChunkNum(kv_id);
            std::sort(kvs.begin(), kvs.end(),
                      [](const KVPairs<Val>& a, const KVPairs<Val>& b) { return a.keys.front() < b.keys.front(); });

            int idx = 0;
            for (const auto& s : kvs) {
                int start = 0;
                for (int i = 0; i < s.keys.size(); ++ i) {
                    if (s.keys[i] == chunk_num-1) {
                        chunks[idx]->resize(s.vals.size()-start);
                        memcpy(chunks[idx]->data(), s.vals.data()+start, (s.vals.size()-start)*sizeof(Val));
                    } else {
                        chunks[idx]->resize(chunk_size);
                        memcpy(chunks[idx]->data(), s.vals.data()+start, chunk_size*sizeof(Val));
                    }
                    start += chunk_size;
                    idx += 1;
                }
            }

            mu_.lock();
            delete recv_kvs_[{kv_id, ts}];
            recv_kvs_.erase({kv_id, ts});
            mu_.unlock();
            if (cb)
                cb();
        });
        SendChunks_(kv_id, ts, false, chunk_ids, std::vector<std::vector<Val>*>(), pos, send_all, local_zero_copy, false, consistency_control);
        return ts;
    }

    /*
     * Pull a list of chunks from server with min clock
     * TODO: Now the API only get the smallest min_clock among all the servers
     *
     * @param kv_id the kv_id users want to handle with
     * @param chunk_ids The chunk_ids that needs to be pulled
     * @param chunks The chunks provided must be created beforehand
     * @param min_clock The pointer to min clock
     * @param send_all whether needs to send to all servers
     * @param local_zero_copy enable local zero copy or not
     * @param cb callback function
     */
    template<typename Val>
    int PullChunksWithMinClock(int kv_id, const std::vector<size_t>& chunk_ids, const std::vector<std::vector<Val>*>& chunks, int* min_clock,
             bool send_all = true, bool local_zero_copy = true, const Callback& cb = nullptr) {
        assert(chunk_ids.size() == chunks.size());
        // 1. partition
        std::vector<size_t> pos = PartitionChunks_(kv_id, chunk_ids);
        // 2. get ts
        int ts = GetTimestampChunk_(kv_id, pos, send_all);
        AddCallback(kv_id, ts, [this, kv_id, ts, chunk_ids, chunks, min_clock, cb]() {
            mu_.lock();
            auto& kvs = static_cast<RecvKVPairsWithMinClock<Val>*>(recv_kvs_[{kv_id, ts}])->recv_kvs;
            mu_.unlock();

            size_t chunk_size = RangeManager::Get().GetChunkSize(kv_id);
            size_t chunk_num = RangeManager::Get().GetChunkNum(kv_id);
            std::sort(kvs.begin(), kvs.end(),
                      [](const std::pair<KVPairs<Val>, int>& a, const std::pair<KVPairs<Val>, int>& b) { return a.first.keys.front() < b.first.keys.front(); });

            int idx = 0;
            int min_clock_local = std::numeric_limits<int>::max();
            for (const auto& s : kvs) {
                int start = 0;
                for (int i = 0; i < s.first.keys.size(); ++ i) {
                    if (s.first.keys[i] == chunk_num-1) {
                        chunks[idx]->resize(s.first.vals.size()-start);
                        memcpy(chunks[idx]->data(), s.first.vals.data()+start, (s.first.vals.size()-start)*sizeof(Val));
                    } else {
                        chunks[idx]->resize(chunk_size);
                        memcpy(chunks[idx]->data(), s.first.vals.data()+start, chunk_size*sizeof(Val));
                    }
                    start += chunk_size;
                    idx += 1;
                }
                if (s.second < min_clock_local)
                    min_clock_local = s.second;
            }
            *min_clock = min_clock_local;

            mu_.lock();
            delete recv_kvs_[{kv_id, ts}];
            recv_kvs_.erase({kv_id, ts});
            mu_.unlock();
            if (cb)
                cb();
        });
        bool with_min_clock = true;
        SendChunks_(kv_id, ts, false, chunk_ids, std::vector<std::vector<Val>*>(), pos, send_all, local_zero_copy, with_min_clock, true);
        return ts;
    }

    // Deprecated
    template<typename Val>
    int PushLocal(int kv_id, int dst, const std::vector<husky::constants::Key>& keys, 
            const std::vector<Val>& vals, const Callback& cb = nullptr) {
        int ts = customer_->NewRequest(kv_id, 1);
        AddCallback(kv_id, ts, cb);
        husky::base::BinStream bin;
        bool isPush = true;
        int cmd = 0;
        int src = info_.global_id;
        bin << kv_id << ts << cmd << isPush << src;
        bin << keys << vals;
        customer_->send(info_.get_tid(dst), bin);
        return ts;
    }

    // Deprecated
    template<typename Val>
    int PullLocal(int kv_id, int dst, const std::vector<husky::constants::Key>& keys, 
            std::vector<Val>* vals, const Callback& cb = nullptr) {
        int ts = customer_->NewRequest(kv_id, 1);
        AddCallback(kv_id, ts, [this, kv_id, ts, vals, cb]() mutable {
            mu_.lock();
            auto& kvs = static_cast<RecvKVPairs<Val>*>(recv_kvs_[{kv_id, ts}])->recv_kvs;
            mu_.unlock();

            assert(kvs.size() == 1);
            // Assume the kvs are ordered
            vals->resize(kvs[0].vals.size());
            for (size_t i = 0; i < vals->size(); ++ i) {
                (*vals)[i] = std::move(kvs[0].vals[i]);
            }

            mu_.lock();
            delete recv_kvs_[{kv_id, ts}];
            recv_kvs_.erase({kv_id, ts});
            mu_.unlock();
            if (cb) cb();
        });
        husky::base::BinStream bin;
        bool isPush = false;
        int cmd = 0;
        int src = info_.global_id;
        bin << kv_id << ts << cmd << isPush << src;
        bin << keys;
        customer_->send(info_.get_tid(dst), bin);
        return ts;
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

   private:
    /*
     * \brief UniqueProcess for every individual kvstore
     */
    template <typename Val>
    void UniqueProcess(int kv_id, int ts, husky::base::BinStream& bin, bool runCallback) {
        int cmd;
        bool push;  // push or not
        int src;
        bin >> cmd;
        bin >> push;
        bin >> src;
        if (push == true)
            ;                      // if is push
        else if (push == false) {  // if is pull
            auto update_kvs = [this, kv_id, ts](const KVPairs<Val>& kvs) {
                mu_.lock();
                if (recv_kvs_.find({kv_id, ts}) == recv_kvs_.end()) {
                    recv_kvs_[{kv_id, ts}] = new RecvKVPairs<Val>();
                }
                // only push non-empty size
                // husky::LOG_I << RED("pull size: "+std::to_string(kvs.keys.size()));
                if (kvs.keys.size() != 0) {
                    static_cast<RecvKVPairs<Val>*>(recv_kvs_[{kv_id, ts}])->recv_kvs.push_back(kvs);
                }
                mu_.unlock();
            };
            auto update_kvs_with_min_clock = [this, kv_id, ts](const std::pair<KVPairs<Val>, int>& kvs) {
                mu_.lock();
                if (recv_kvs_.find({kv_id, ts}) == recv_kvs_.end()) {
                    recv_kvs_[{kv_id, ts}] = new RecvKVPairsWithMinClock<Val>();
                }
                // only push non-empty size
                // husky::LOG_I << RED("pull size: "+std::to_string(kvs.first.keys.size()) + " kv, ts: "+std::to_string(kv_id)+" "+std::to_string(ts));
                if (kvs.first.keys.size() != 0) {
                    static_cast<RecvKVPairsWithMinClock<Val>*>(recv_kvs_[{kv_id, ts}])->recv_kvs.push_back(kvs);
                }
                mu_.unlock();
            };
            cmd %= consistency_control_off_magic_;
            if (cmd == 2) {  // zero-copy enabled
                // husky::LOG_I << RED("zero-copy in Pull is enabled");
                std::uintptr_t ptr;
                bin >> ptr;
                auto* p_recv = reinterpret_cast<KVPairs<Val>*>(ptr);
                update_kvs(*p_recv);
                delete p_recv;
            } else if (cmd == 13 || cmd == 11) {  // for PullChunksWithMinClock
                std::pair<KVPairs<Val>, int> kvs;
                bin >> kvs.first.keys >> kvs.first.vals >> kvs.second;
                update_kvs_with_min_clock(kvs);
            } else {
                // husky::LOG_I << RED("zero-copy in Pull is disabled");
                KVPairs<Val> kvs;
                // Format: keys, values
                bin >> kvs.keys >> kvs.vals;
                update_kvs(kvs);
            }
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

    template <typename Val, typename C>
    int Pull_(int kv_id, const pslite::SArray<husky::constants::Key>& keys, C* vals, bool send_all, bool local_zero_copy, bool consistency_control, const Callback& cb) {
        // 1. slice the message
        KVPairs<Val> kvs;
        kvs.keys = keys;
        SlicedKVs<Val> sliced;
        Slice_(kvs, RangeManager::Get().GetServerKeyRanges(kv_id), &sliced);
        // 2. get ts
        int ts = GetTimestamp_(kv_id, sliced);
        // 3. add callback
        AddCallback(kv_id, ts, [this, kv_id, ts, keys, vals, cb]() mutable {
            mu_.lock();
            auto& kvs = static_cast<RecvKVPairs<Val>*>(recv_kvs_[{kv_id, ts}])->recv_kvs;
            mu_.unlock();

            // do check
            size_t total_key = 0, total_val = 0;
            for (const auto& s : kvs) {
                pslite::Range range = pslite::FindRange(keys, s.keys.front(), s.keys.back() + 1);
                total_key += s.keys.size();
                total_val += s.vals.size();
            }

            // fill vals and lens
            std::sort(kvs.begin(), kvs.end(),
                      [](const KVPairs<Val>& a, const KVPairs<Val>& b) { return a.keys.front() < b.keys.front(); });
            // resize, do we need to really free the memory?
            vals->resize(total_val);
            Val* p_vals = vals->data();
            for (const auto& s : kvs) {
                memcpy(p_vals, s.vals.data(), s.vals.size() * sizeof(Val));
                p_vals += s.vals.size();
            }

            mu_.lock();
            delete recv_kvs_[{kv_id, ts}];
            recv_kvs_.erase({kv_id, ts});
            mu_.unlock();
            if (cb)
                cb();
        });
        // 4. send
        Send_(kv_id, ts, false, sliced, send_all, local_zero_copy, consistency_control);
        return ts;
    }

    /*
     * 1. The Slice function to slice the parameters for Push/Pull, stored in SlicedKVs<Val>
     */
    template <typename Val>
    void Slice_(const KVPairs<Val>& send, const std::vector<pslite::Range>& ranges, SlicedKVs<Val>* sliced) {
        sliced->resize(ranges.size());

        // find the positions in msg.key
        size_t n = ranges.size();
        std::vector<size_t> pos(n + 1);
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
            pos[i + 1] = pos[i] + len;

            // don't send it to severs for empty kv
            sliced->at(i).first = (len != 0);
        }
        if (send.keys.empty())
            return;

        // the length of value
        size_t k = 0, val_begin = 0, val_end = 0;
        k = send.vals.size() / send.keys.size();

        // slice
        for (size_t i = 0; i < n; ++i) {
            if (pos[i + 1] == pos[i]) {
                sliced->at(i).first = false;
                continue;
            }
            sliced->at(i).first = true;
            auto& kv = sliced->at(i).second;
            kv.keys = send.keys.segment(pos[i], pos[i + 1]);
            kv.vals = send.vals.segment(pos[i] * k, pos[i + 1] * k);
        }
    }

    /*
     * 2. GetTimestamp_ function to make NewRequest to customer_ and return ts for Push/Pull
     *
     * Identify wait_num
     */
    template<typename Val>
    int GetTimestamp_(int kv_id, const SlicedKVs<Val>& sliced) { 
        int wait_num = 0;
        for (size_t i = 0; i < sliced.size(); ++ i) {
            if (sliced[i].first) wait_num += 1;
        }
        int ts = customer_->NewRequest(kv_id, wait_num);
        // husky::LOG_I << RED("Wait num: "+std::to_string(wait_num));
        return ts;
    }


    /*
     * 3. The Send_ function to send out Push/Pull request
     *
     * @return ts 
     */
    template <typename Val>
    void Send_(int kv_id, int ts, bool push, const SlicedKVs<Val>& sliced, bool send_all, bool local_zero_copy, bool consistency_control) {
        int src = info_.global_id;
        int cmd = 0;  // cmd 0 for normal
        for (size_t i = 0; i < sliced.size(); ++i) {
            if (!send_all && !sliced[i].first) {  // if no need to send all, skip empty sliced
                continue;
            }
            husky::base::BinStream bin;
            bin << kv_id << ts;
            if (local_zero_copy == true && info_.local_server_ids.find(i) != info_.local_server_ids.end()) {  // if enable local_zero_copy
                if (consistency_control) {
                    bin << cmd + local_zero_copy_magic_;  // 2
                } else {
                    bin << cmd + local_zero_copy_magic_ + consistency_control_off_magic_;  // 102
                }
                bin << push << src;
                if (sliced[i].first) {  // if no empty, don't send the size
                    auto& kvs = sliced[i].second;
                    // husky::LOG_I << RED("zero-copy enable");
                    KVPairs<Val>* p = new KVPairs<Val>();  // delete by server
                    p->keys = kvs.keys;
                    p->vals = kvs.vals;
                    bin << reinterpret_cast<std::uintptr_t>(p);
                }
            } else {
                if (consistency_control) {
                    bin << cmd;  // 0
                } else {
                    bin << cmd + consistency_control_off_magic_;  // 100
                }
                bin << push << src;
                if (sliced[i].first) {  // if no empty, don't send the size
                    auto& kvs = sliced[i].second;
                    if (push)
                        bin << kvs.keys << kvs.vals;
                    else
                        bin << kvs.keys;
                }
            }
            // husky::LOG_I << CLAY("sending to "+std::to_string(i)+" size: "+std::to_string(bin.size()));
            customer_->send(info_.get_tid(i), bin);
        }
    }


    /*
     * 1. PartitionChunks_ function to partition the provided chunk_ids for PushChunks/PullChunks
     */
    std::vector<size_t> PartitionChunks_(int kv_id, const std::vector<size_t>& chunk_ids) {
        // partition the chunk_ids
        const std::vector<pslite::Range>& ranges = RangeManager::Get().GetServerKeyRanges(kv_id);
        int chunk_size = RangeManager::Get().GetChunkSize(kv_id);
        size_t n = ranges.size();
        int range_id = 0;
        std::vector<size_t> pos(n + 1);
        pos[0] = 0;
        int pos_id = 1;
        for (size_t i = 0; i < chunk_ids.size(); ++ i) {
            while (chunk_ids[i]*chunk_size >= ranges[range_id].end()) {
                pos[pos_id] = i;
                pos_id += 1;
                range_id += 1;
                assert(range_id < n);
            }
        }
        for (size_t i = pos_id; i < n+1; ++ i) {
            pos[i] = chunk_ids.size();
        }
        return pos;
    }

    /*
     * 2. GetTimestampChunk_ function to make NewRequest to customer_ and return ts for PushChunks/PushChunks
     *
     * Identify wait_num
     */
    int GetTimestampChunk_(int kv_id, const std::vector<size_t>& pos, bool send_all) {
        int wait_num = 0;
        for (int i = 0; i < pos.size() - 1; ++ i) {
            if (pos[i] != pos[i+1]) wait_num += 1;
        }
        // husky::LOG_I << RED("wait num: "+std::to_string(wait_num));
        int ts = customer_->NewRequest(kv_id, wait_num);
        // husky::LOG_I << RED("Wait num: "+std::to_string(wait_num));
        return ts;
    }

    /*
     * 3. SendChunks_ function for PushChunks and PullChunks
     *
     * Used for PushChunks and PullChunks
     */
    template<typename Val>
    void SendChunks_(int kv_id, int ts, bool push,
            const std::vector<size_t>& chunk_ids, const std::vector<std::vector<Val>*>& chunks, 
            const std::vector<size_t>& pos, bool send_all, bool local_zero_copy, bool with_min_clock, bool consistency_control) {
        const std::vector<pslite::Range>& ranges = RangeManager::Get().GetServerKeyRanges(kv_id);
        size_t n = ranges.size();

        // send
        int src = info_.global_id;
        int cmd = 1;  // cmd 1 for chunk push
        for (size_t i = 0; i < n; ++ i) {
            if (!send_all && pos[i] == pos[i+1]) {  // if no need to send all, skip empty sliced
                continue;
            }
            husky::base::BinStream bin;
            bin << kv_id << ts;
            if (local_zero_copy == true && info_.local_server_ids.find(i) != info_.local_server_ids.end()) {
                int local_cmd = cmd + local_zero_copy_magic_;  // 3
                if (with_min_clock)
                    local_cmd += with_min_clock_magic_;  // 13
                if (consistency_control == false)
                    local_cmd += consistency_control_off_magic_;  // 103
                if (with_min_clock == true && consistency_control == false) {
                    throw husky::base::HuskyException("with_min_clock and consistency_control_off cannot be enable at the same time");
                }
                // 3, 13, 103
                bin << local_cmd << push << src;
                if (pos[i] != pos[i+1]) {  // if empty, don't send the size
                    auto* p = new std::pair<std::vector<size_t>, std::vector<std::vector<Val>>>();
                    // TODO, still need to copy once
                    p->first.reserve(pos[i+1]-pos[i]);
                    for (size_t j = pos[i]; j < pos[i+1]; ++ j) {  // chunks_ids
                        p->first.push_back(chunk_ids[j]);
                    }
                    if (push) {
                        p->second.reserve(pos[i+1]-pos[i]);
                        for (size_t j = pos[i]; j < pos[i+1]; ++ j) {  // chunks
                            p->second.push_back(*chunks[j]);
                        }
                    }
                    bin << reinterpret_cast<std::uintptr_t>(p);
                }
            } else {
                int local_cmd = cmd;  // 1
                if (with_min_clock)
                    local_cmd += with_min_clock_magic_;  // 11
                if (consistency_control == false)
                    local_cmd += consistency_control_off_magic_;  // 101
                if (with_min_clock == true && consistency_control == false) {
                    throw husky::base::HuskyException("with_min_clock and consistency_control_off cannot be enable at the same time");
                }
                // 1, 11, 101
                bin << local_cmd << push << src;
                if (pos[i] != pos[i+1]) {  // if empty, don't send the size
                    bin << static_cast<size_t>(pos[i+1]-pos[i]);
                    for (size_t j = pos[i]; j < pos[i+1]; ++ j) {  // chunk_ids
                        bin << chunk_ids[j];
                    }
                    if (push) {
                        for (size_t j = pos[i]; j < pos[i+1]; ++ j) {  // chunks
                            bin << *chunks[j];
                        }
                    }
                }
            }
            customer_->send(info_.get_tid(i), bin);
        }
    }

   private:
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

    static const int consistency_control_off_magic_ = 100;
    static const int local_zero_copy_magic_ = 2;
    static const int with_min_clock_magic_ = 10;
};

}  // namespace kvstore
