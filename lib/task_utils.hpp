#pragma once

#include "core/constants.hpp"
#include "core/task.hpp"
#include "kvstore/kvstore.hpp"

namespace husky {
namespace {

/*
 * This function is used to create a kvstore for a task according to the hint provided
 */
int create_kvstore_and_set_hint(const std::map<std::string, std::string>& hint, MLTask& task) {
    int kv = -1;
    try {
        task.set_hint(hint);
        kv = kvstore::KVStore::Get().CreateKVStore<float>(hint);
        task.set_kvstore(kv);
        husky::LOG_I << GREEN("Set to " + hint.at(husky::constants::kType));
    } catch (...) {
        throw base::HuskyException("Unknown hint");
    }
    return kv;
}

}  // namespace anonymous
}  // namespace husky
