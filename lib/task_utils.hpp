#pragma once

#include "core/constants.hpp"
#include "core/task.hpp"
#include "kvstore/kvstore.hpp"

#include "core/utility.hpp"
#include "husky/core/context.hpp"

namespace husky {
namespace {

/*
 * This function is used to create a kvstore for a task according to the hint provided
 */
int create_kvstore_and_set_hint(const std::map<std::string, std::string>& hint, MLTask& task, size_t max_key) {
    int kv = -1;
    try {
        task.set_hint(hint);
        kv = kvstore::KVStore::Get().CreateKVStore<float>(hint, max_key);
        task.set_kvstore(kv);
        if (Context::get_process_id() == 0)
            husky::LOG_I << GREEN("Set to " + hint.at(husky::constants::kType));
    } catch (...) {
        utility::print_hint(hint);
        throw base::HuskyException("task_utils.hpp: Unknown hint");
    }
    return kv;
}

}  // namespace anonymous
}  // namespace husky
