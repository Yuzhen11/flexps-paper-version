#pragma once

#include <unordered_map>
#include "husky/base/serialization.hpp"
#include "kvstore/kvpairs.hpp"

namespace kvstore {
namespace {

// update function for push
template<typename Val>
static void update(husky::base::BinStream& bin, std::unordered_map<int, Val>& store) {
    KVPairs<Val> recv;
    bin >> recv.keys >> recv.vals;
    for (int i = 0; i < recv.keys.size(); ++ i) {
        store[recv.keys[i]] += recv.vals[i];
    }
}

// assign function for push
template<typename Val>
static void assign(husky::base::BinStream& bin, std::unordered_map<int, Val>& store) {
    KVPairs<Val> recv;
    bin >> recv.keys >> recv.vals;
    for (int i = 0; i < recv.keys.size(); ++ i) {
        store[recv.keys[i]] = recv.vals[i];
    }
}

// retrieve function to retrieve the valued indexed by key
template<typename Val>
static KVPairs<Val> retrieve(husky::base::BinStream& bin, std::unordered_map<int, Val>& store) {
    KVPairs<Val> recv;
    KVPairs<Val> send;
    bin >> recv.keys >> recv.vals;
    send.keys = recv.keys;
    send.vals.resize(recv.keys.size());
    for (int i = 0; i < send.keys.size(); ++ i) {
        send.vals[i] = store[send.keys[i]];
    }
    return send;
}

}  // namespace
}  // namespace kvstore
