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

template<typename Val, typename StorageT>
void UpdateStore(int kv_id, int server_id, StorageT& store, size_t key, Val& value, bool is_vector = false, bool is_assign = true) {
    int interval = RangeManager::Get().GetServerInterval(kv_id, server_id);

    if (is_vector) {
        if (is_assign) {
            store[key - interval] = value;
        } else {
            store[key - interval] += value;
        }
    } else {
        if (is_assign) {
            store[key] = value;
        } else {
            store[key] += value;
        }
    }
}

// update function for push
template <typename Val, typename StorageT>
void update(int kv_id, int server_id, husky::base::BinStream& bin, StorageT& store, int cmd, bool is_vector = false, bool is_assign = true) {
    if (cmd == 0) {
        KVPairs<Val> recv;
        bin >> recv.keys >> recv.vals;
        for (size_t i = 0; i < recv.keys.size(); ++i) {
            UpdateStore<Val, StorageT>(kv_id, server_id, store, recv.keys[i], recv.vals[i], is_vector, is_assign);
        }
    } else if (cmd == 1) {
        size_t chunk_size = RangeManager::Get().GetChunkSize(kv_id);
        std::vector<size_t> chunk_ids;
        bin >> chunk_ids;
        for (auto chunk_id : chunk_ids) {
            size_t start_id = chunk_id * chunk_size;
            std::vector<Val> chunk;
            bin >> chunk;
            for (size_t i = 0; i < chunk.size(); ++ i) {
                UpdateStore<Val, StorageT>(kv_id, server_id, store, start_id + i, chunk[i], is_vector, is_assign);
            }
        }
        // husky::LOG_I << RED("Done");
    } else if (cmd == 2) {  // enable zero-copy
        std::uintptr_t ptr;
        bin >> ptr;
        auto* p_recv = reinterpret_cast<KVPairs<Val>*>(ptr);
        for (size_t i = 0; i < p_recv->keys.size(); ++ i) {
            UpdateStore<Val, StorageT>(kv_id, server_id, store, p_recv->keys[i], p_recv->vals[i], is_vector, is_assign);
        }
        delete p_recv;
    } else if (cmd == 3) {  // zero-copy chunks
        size_t chunk_size = RangeManager::Get().GetChunkSize(kv_id);
        std::uintptr_t ptr;
        bin >> ptr;
        auto* p_recv = reinterpret_cast<std::pair<std::vector<size_t>, std::vector<std::vector<Val>>>*>(ptr);
        auto& chunk_ids = p_recv->first;
        auto& chunks = p_recv->second;
        for (size_t i = 0; i < chunk_ids.size(); ++ i) {
            size_t start_id = chunk_ids[i] * chunk_size;
            for (size_t j = 0; j < chunks[i].size(); ++ j) {
                UpdateStore<Val, StorageT>(kv_id, server_id, store, start_id + j, chunks[i][j], is_vector, is_assign);
            }
        }
        delete p_recv;
    } else {
        throw husky::base::HuskyException("Unknown cmd");
    }
}

template<typename Val, typename StorageT>
Val RetrieveStore(int kv_id, int server_id, StorageT& store, size_t key, bool is_vector = false) {
    Val value;

    int interval = RangeManager::Get().GetServerInterval(kv_id, server_id);

    if (is_vector) {
        value = store[key - interval];
    } else {
        value = store[key];
    }

    return value;
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
        for (size_t i = 0; i < send.keys.size(); ++i) {
            send.vals[i] = RetrieveStore<Val, StorageT>(kv_id, server_id, store, send.keys[i], is_vector);
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
        for (auto chunk_id : chunk_ids) {
            send.keys.push_back(chunk_id);
            // husky::LOG_I << RED("retrieve chunk_id " + std::to_string(chunk_id));
            size_t start_id = chunk_id * chunk_size;
            int real_chunk_size = chunk_size;
            if (chunk_id == chunk_num-1) 
                real_chunk_size = RangeManager::Get().GetLastChunkSize(kv_id);
            for (int i = 0; i < real_chunk_size; ++ i) {
                send.vals.push_back(RetrieveStore<Val, StorageT>(kv_id, server_id, store, start_id + i, is_vector));
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
        for (size_t i = 0; i < send.keys.size(); ++ i) {
            send.vals[i] = RetrieveStore<Val, StorageT>(kv_id, server_id, store, send.keys[i], is_vector);
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
        for (auto chunk_id : chunk_ids) {
            send.keys.push_back(chunk_id);
            // husky::LOG_I << RED("retrieve chunk_id " + std::to_string(chunk_id));
            size_t start_id = chunk_id * chunk_size;
            int real_chunk_size = chunk_size;
            if (chunk_id == chunk_num-1) 
                real_chunk_size = RangeManager::Get().GetLastChunkSize(kv_id);
            for (int i = 0; i < real_chunk_size; ++ i) {
                send.vals.push_back(RetrieveStore<Val, StorageT>(kv_id, server_id, store, start_id + i, is_vector));
            }
        }
        return send;
    } else {
        throw husky::base::HuskyException("Unknown cmd");
    }
}

}  // namespace
}  // namespace kvstore
