#pragma once

#include <unordered_map>
#include "core/constants.hpp"
#include "husky/base/serialization.hpp"
#include "husky/base/exception.hpp"
#include "kvstore/kvpairs.hpp"
#include "kvstore/range_manager.hpp"
#include "core/color.hpp"

namespace kvstore {
namespace {

// update function for push
template <typename Val, typename StorageT>
void update(int kv_id, int server_id, husky::base::BinStream& bin, StorageT& store, int cmd, bool is_vector = false, bool is_assign = true) {
    if (cmd == 0) {
        KVPairs<Val> recv;
        bin >> recv.keys >> recv.vals;
        int interval = is_vector ? RangeManager::Get().GetServerInterval(kv_id, server_id) : 0;
        if (is_assign) {
            for (size_t i = 0; i < recv.keys.size(); ++ i) {
                store[recv.keys[i] - interval] = recv.vals[i];
            }
        } else {
            for (size_t i = 0; i < recv.keys.size(); ++ i) {
                store[recv.keys[i] - interval] += recv.vals[i];
            }
        }
    } else if (cmd == 1) {
        size_t chunk_size = RangeManager::Get().GetChunkSize(kv_id);
        std::vector<size_t> chunk_ids;
        bin >> chunk_ids;
        int interval = is_vector ? RangeManager::Get().GetServerInterval(kv_id, server_id) : 0;
        for (auto chunk_id : chunk_ids) {
            size_t start_id = chunk_id * chunk_size;
            std::vector<Val> chunk;
            bin >> chunk;
            if (is_assign) {
                for (size_t i = 0; i < chunk.size(); ++i) {
                    store[start_id + i - interval] = chunk[i];
                }
            } else {
                for (size_t i = 0; i < chunk.size(); ++i) {
                    store[start_id + i - interval] += chunk[i];
                }
            }
        }
        // husky::LOG_I << RED("Done");
    } else if (cmd == 2) {  // enable zero-copy
        std::uintptr_t ptr;
        bin >> ptr;
        auto* p_recv = reinterpret_cast<KVPairs<Val>*>(ptr);
        
        int interval = is_vector ? RangeManager::Get().GetServerInterval(kv_id, server_id) : 0;
        if (is_assign) {
            for (size_t i = 0; i < p_recv->keys.size(); ++ i) {
                store[p_recv->keys[i] - interval] = p_recv->vals[i];
            }
        } else {
            for (size_t i = 0; i < p_recv->keys.size(); ++ i) {
                store[p_recv->keys[i] - interval] += p_recv->vals[i];
            }
        }
        
        delete p_recv;
    } else if (cmd == 3) {  // zero-copy chunks
        size_t chunk_size = RangeManager::Get().GetChunkSize(kv_id);
        std::uintptr_t ptr;
        bin >> ptr;
        auto* p_recv = reinterpret_cast<std::pair<std::vector<size_t>, std::vector<std::vector<Val>>>*>(ptr);
        auto& chunk_ids = p_recv->first;
        auto& chunks = p_recv->second;
        int interval = is_vector ? RangeManager::Get().GetServerInterval(kv_id, server_id) : 0;
        for (size_t i = 0; i < chunk_ids.size(); ++ i) {
            size_t start_id = chunk_ids[i] * chunk_size;
            if (is_assign) {
                for (size_t j = 0; j < chunks[i].size(); ++j) {
                    store[start_id + j - interval] = chunks[i][j];
                }
            } else {
                for (size_t j = 0; j < chunks[i].size(); ++j) {
                    store[start_id + j - interval] += chunks[i][j];
                }
            }
        }
        delete p_recv;
    } else {
        throw husky::base::HuskyException("Unknown cmd " + std::to_string(cmd));
    }
}

// retrieve function to retrieve the valued indexed by key
template<typename Val, typename StorageT>
KVPairs<Val> retrieve(int kv_id, int server_id, husky::base::BinStream& bin, StorageT& store, int cmd, bool is_vector = false) {
    if (cmd == 0) {
        KVPairs<Val> recv;
        KVPairs<Val> send;
        bin >> recv.keys;
        send.keys = recv.keys;
        send.vals.resize(recv.keys.size());
        int interval = is_vector ? RangeManager::Get().GetServerInterval(kv_id, server_id) : 0;
        for (size_t i = 0; i < send.keys.size(); ++i) {
            send.vals[i] = store[send.keys[i] - interval];
        }
        return send;
    } else if (cmd == 1) {
        size_t chunk_size = RangeManager::Get().GetChunkSize(kv_id);
        size_t chunk_num = RangeManager::Get().GetChunkNum(kv_id);
        std::vector<size_t> chunk_ids;
        bin >> chunk_ids;
        KVPairs<Val> send;
        send.keys.reserve(chunk_ids.size());
        send.vals.reserve(chunk_ids.size()*chunk_size);
        int interval = is_vector ? RangeManager::Get().GetServerInterval(kv_id, server_id) : 0;
        for (auto chunk_id : chunk_ids) {
            send.keys.push_back(chunk_id);
            // husky::LOG_I << RED("retrieve chunk_id " + std::to_string(chunk_id));
            size_t start_id = chunk_id * chunk_size;
            int real_chunk_size = chunk_size;
            if (chunk_id == chunk_num-1) 
                real_chunk_size = RangeManager::Get().GetLastChunkSize(kv_id);
            for (int i = 0; i < real_chunk_size; ++ i) {
                send.vals.push_back(store[start_id + i - interval]);
            }
        }
        return send;
    } else if (cmd == 2) {  // enable zero-copy
        KVPairs<Val> send;
        std::uintptr_t ptr;
        bin >> ptr;
        auto* p_recv = reinterpret_cast<KVPairs<Val>*>(ptr);
        send.keys = p_recv->keys;
        send.vals.resize(p_recv->keys.size());
        int interval = is_vector ? RangeManager::Get().GetServerInterval(kv_id, server_id) : 0;
        for (size_t i = 0; i < send.keys.size(); ++ i) {
            send.vals[i] = store[send.keys[i] - interval];
        }
        delete p_recv;
        return send;
    } else if (cmd == 3) {
        size_t chunk_size = RangeManager::Get().GetChunkSize(kv_id);
        size_t chunk_num = RangeManager::Get().GetChunkNum(kv_id);
        std::uintptr_t ptr;
        bin >> ptr;
        auto* p_recv = reinterpret_cast<std::pair<std::vector<size_t>, std::vector<std::vector<Val>>>*>(ptr);
        auto& chunk_ids = p_recv->first;
        KVPairs<Val> send;
        send.keys.reserve(chunk_ids.size());
        send.vals.reserve(chunk_ids.size()*chunk_size);
        int interval = is_vector ? RangeManager::Get().GetServerInterval(kv_id, server_id) : 0;
        for (auto chunk_id : chunk_ids) {
            send.keys.push_back(chunk_id);
            // husky::LOG_I << RED("retrieve chunk_id " + std::to_string(chunk_id));
            size_t start_id = chunk_id * chunk_size;
            int real_chunk_size = chunk_size;
            if (chunk_id == chunk_num-1) 
                real_chunk_size = RangeManager::Get().GetLastChunkSize(kv_id);
            for (int i = 0; i < real_chunk_size; ++ i) {
                send.vals.push_back(store[start_id + i - interval]);
            }
        }
        return send;
    } else {
        throw husky::base::HuskyException("Unknown cmd " + std::to_string(cmd));
    }
}

}  // namespace
}  // namespace kvstore
