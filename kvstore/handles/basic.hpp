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
template <typename Val>
void update(int kv_id, husky::base::BinStream& bin, std::unordered_map<husky::constants::Key, Val>& store, int cmd) {
    if (cmd == 0) {
        KVPairs<Val> recv;
        bin >> recv.keys >> recv.vals;
        for (size_t i = 0; i < recv.keys.size(); ++i) {
            store[recv.keys[i]] += recv.vals[i];
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
                // husky::LOG_I << RED("add: "+std::to_string(start_id+i));
                store[start_id + i] += chunk[i];
            }
        }
        // husky::LOG_I << RED("Done");
    } else if (cmd == 2) {  // enable zero-copy
        std::uintptr_t ptr;
        bin >> ptr;
        auto* p_recv = reinterpret_cast<KVPairs<Val>*>(ptr);
        for (size_t i = 0; i < p_recv->keys.size(); ++ i) {
            store[p_recv->keys[i]] += p_recv->vals[i];
            // husky::LOG_I << RED("Assign: "+std::to_string(p_recv->keys[i])+" "+std::to_string(p_recv->vals[i]));
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
                store[start_id + j] += chunks[i][j];
            }
        }
        delete p_recv;
    } else {
        throw husky::base::HuskyException("Unknown cmd");
    }
}

// update function for push
template <typename Val>
void update(int kv_id, husky::base::BinStream& bin, std::vector<Val>& store, int cmd, int server_id) {
    assert(store.size() >= RangeManager::Get().GetServerSize(kv_id, server_id));
    // get interval
    int interval = RangeManager::Get().GetServerInterval(kv_id, server_id);
    if (cmd == 0) {
        KVPairs<Val> recv;
        bin >> recv.keys >> recv.vals;
        for (size_t i = 0; i < recv.keys.size(); ++i) {
            store[recv.keys[i] - interval] += recv.vals[i];
        }
    } else if (cmd == 1) {
        size_t chunk_size = RangeManager::Get().GetChunkSize(kv_id);
        std::vector<size_t> chunk_ids;
        bin >> chunk_ids;
        for (auto chunk_id : chunk_ids) {
            size_t start_id = chunk_id * chunk_size - interval;
            std::vector<Val> chunk;
            bin >> chunk;
            for (int i = 0; i < chunk.size(); ++ i) {
                // husky::LOG_I << RED("add: "+std::to_string(start_id+i));
                store[start_id + i] += chunk[i];
            }
        }
        // husky::LOG_I << RED("Done");
    } else if (cmd == 2) {  // enable zero-copy
        std::uintptr_t ptr;
        bin >> ptr;
        auto* p_recv = reinterpret_cast<KVPairs<Val>*>(ptr);
        for (size_t i = 0; i < p_recv->keys.size(); ++ i) {
            store[p_recv->keys[i] - interval] += p_recv->vals[i];
            // husky::LOG_I << RED("Assign: "+std::to_string(p_recv->keys[i])+" "+std::to_string(p_recv->vals[i]));
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
            size_t start_id = chunk_ids[i] * chunk_size - interval;
            for (size_t j = 0; j < chunks[i].size(); ++ j) {
                store[start_id + j] += chunks[i][j];
            }
        }
        delete p_recv;
    } else {
        throw husky::base::HuskyException("Unknown cmd");
    }
}

// assign function for push
template <typename Val>
void assign(int kv_id, husky::base::BinStream& bin, std::unordered_map<husky::constants::Key, Val>& store, int cmd) {
    if (cmd == 0) {
        KVPairs<Val> recv;
        bin >> recv.keys >> recv.vals;
        for (size_t i = 0; i < recv.keys.size(); ++i) {
            store[recv.keys[i]] = recv.vals[i];
        }
    } else if (cmd == 1) {
        size_t chunk_size = RangeManager::Get().GetChunkSize(kv_id);
        std::vector<size_t> chunk_ids;
        bin >> chunk_ids;
        for (auto chunk_id : chunk_ids) {
            size_t start_id = chunk_id * chunk_size;
            std::vector<Val> chunk;
            bin >> chunk;
            for (int i = 0; i < chunk.size(); ++ i) {
                // husky::LOG_I << RED("add: "+std::to_string(start_id+i));
                store[start_id + i] = chunk[i];
            }
        }
        // husky::LOG_I << RED("Done");
    } else if (cmd == 2) {  // enable zero-copy
        std::uintptr_t ptr;
        bin >> ptr;
        auto* p_recv = reinterpret_cast<KVPairs<Val>*>(ptr);
        for (size_t i = 0; i < p_recv->keys.size(); ++ i) {
            store[p_recv->keys[i]] = p_recv->vals[i];
            // husky::LOG_I << RED("Assign: "+std::to_string(p_recv->keys[i])+" "+std::to_string(p_recv->vals[i]));
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
                store[start_id + j] = chunks[i][j];
            }
        }
        delete p_recv;
    } else {
        throw husky::base::HuskyException("Unknown cmd");
    }
}

// assign function for push
template <typename Val>
void assign(int kv_id, husky::base::BinStream& bin, std::vector<Val>& store, int cmd, int server_id) {
    assert(store.size() >= RangeManager::Get().GetServerSize(kv_id, server_id));
    // get interval
    int interval = RangeManager::Get().GetServerInterval(kv_id, server_id);
    if (cmd == 0) {
        KVPairs<Val> recv;
        bin >> recv.keys >> recv.vals;
        for (size_t i = 0; i < recv.keys.size(); ++i) {
            store[recv.keys[i] - interval] = recv.vals[i];
        }
    } else if (cmd == 1) {
        size_t chunk_size = RangeManager::Get().GetChunkSize(kv_id);
        std::vector<size_t> chunk_ids;
        bin >> chunk_ids;
        for (auto chunk_id : chunk_ids) {
            size_t start_id = chunk_id * chunk_size - interval;
            std::vector<Val> chunk;
            bin >> chunk;
            for (int i = 0; i < chunk.size(); ++ i) {
                // husky::LOG_I << RED("add: "+std::to_string(start_id+i));
                store[start_id + i] = chunk[i];
            }
        }
        // husky::LOG_I << RED("Done");
    } else if (cmd == 2) {  // enable zero-copy
        std::uintptr_t ptr;
        bin >> ptr;
        auto* p_recv = reinterpret_cast<KVPairs<Val>*>(ptr);
        for (size_t i = 0; i < p_recv->keys.size(); ++ i) {
            store[p_recv->keys[i] - interval] = p_recv->vals[i];
            // husky::LOG_I << RED("Assign: "+std::to_string(p_recv->keys[i])+" "+std::to_string(p_recv->vals[i]));
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
            size_t start_id = chunk_ids[i] * chunk_size - interval;
            for (size_t j = 0; j < chunks[i].size(); ++ j) {
                store[start_id + j] = chunks[i][j];
            }
        }
        delete p_recv;
    } else {
        throw husky::base::HuskyException("Unknown cmd");
    }
}

// retrieve function to retrieve the valued indexed by key
template <typename Val>
KVPairs<Val> retrieve(int kv_id, husky::base::BinStream& bin, std::unordered_map<husky::constants::Key, Val>& store, int cmd) {
    if (cmd == 0) {
        KVPairs<Val> recv;
        KVPairs<Val> send;
        bin >> recv.keys;
        send.keys = recv.keys;
        send.vals.resize(recv.keys.size());
        for (size_t i = 0; i < send.keys.size(); ++i) {
            send.vals[i] = store[send.keys[i]];
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
                send.vals.push_back(store[start_id + i]);
                // husky::LOG_I << RED("val: "+std::to_string(store[start_id + i]));
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
            send.vals[i] = store[send.keys[i]];
            // husky::LOG_I << RED("Retrieve: "+std::to_string(send.keys[i])+" "+std::to_string(send.vals[i]));
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
                send.vals.push_back(store[start_id + i]);
                // husky::LOG_I << RED("val: "+std::to_string(store[start_id + i]));
            }
        }
        return send;
    } else {
        throw husky::base::HuskyException("Unknown cmd");
    }
}

// retrieve function to retrieve the valued indexed by key
template <typename Val>
KVPairs<Val> retrieve(int kv_id, husky::base::BinStream& bin, std::vector<Val>& store, int cmd, int server_id) {
    assert(store.size() >= RangeManager::Get().GetServerSize(kv_id, server_id));
    // get interval
    int interval = RangeManager::Get().GetServerInterval(kv_id, server_id);
    if (cmd == 0) {
        KVPairs<Val> recv;
        KVPairs<Val> send;
        bin >> recv.keys;
        send.keys = recv.keys;
        send.vals.resize(recv.keys.size());
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
        for (auto chunk_id : chunk_ids) {
            send.keys.push_back(chunk_id);
            // husky::LOG_I << RED("retrieve chunk_id " + std::to_string(chunk_id));
            size_t start_id = chunk_id * chunk_size - interval;
            int real_chunk_size = chunk_size;
            if (chunk_id == chunk_num-1) 
                real_chunk_size = RangeManager::Get().GetLastChunkSize(kv_id);
            for (int i = 0; i < real_chunk_size; ++ i) {
                send.vals.push_back(store[start_id + i]);
                // husky::LOG_I << RED("val: "+std::to_string(store[start_id + i]));
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
            send.vals[i] = store[send.keys[i] - interval];
            // husky::LOG_I << RED("Retrieve: "+std::to_string(send.keys[i])+" "+std::to_string(send.vals[i]));
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
            size_t start_id = chunk_id * chunk_size - interval;
            int real_chunk_size = chunk_size;
            if (chunk_id == chunk_num-1) 
                real_chunk_size = RangeManager::Get().GetLastChunkSize(kv_id);
            for (int i = 0; i < real_chunk_size; ++ i) {
                send.vals.push_back(store[start_id + i]);
                // husky::LOG_I << RED("val: "+std::to_string(store[start_id + i]));
            }
        }
        return send;
    } else {
        throw husky::base::HuskyException("Unknown cmd");
    }
}

}  // namespace
}  // namespace kvstore
