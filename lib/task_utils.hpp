#pragma once

#include "core/task.hpp"
#include "kvstore/kvstore.hpp"

namespace husky {
namespace {

/*
 * This function is used to create a kvstore for a task according to the hint provided
 */
int create_kvstore_and_set_hint(const std::string& hint, MLTask& task, int num_train_workers) {
    std::vector<std::string> instructions;
    boost::split(instructions, hint, boost::is_any_of(":"));
    int kv = -1;
    try {
        task.set_hint(hint);
        std::string& first = instructions.at(0);
        if (first == "PS") {
            std::string& second = instructions.at(1);
            if (second == "BSP") {
                kv = kvstore::KVStore::Get().CreateKVStore<float>(kvstore::KVServerBSPHandle<float>(num_train_workers));
            } else if (second == "SSP") {
                int staleness;
                if (instructions.size() == 2) staleness = 1;
                else if (instructions.at(2) == "SSPWorker") {
                    staleness = std::stoi(instructions.at(3));
                } else {
                    throw;
                }
                kv = kvstore::KVStore::Get().CreateKVStore<float>(
                    kvstore::KVServerSSPHandle<float>(num_train_workers, staleness));
            } else if (second == "ASP") {
                kv = kvstore::KVStore::Get().CreateKVStore<float>(
                    kvstore::KVServerDefaultAddHandle<float>());  // use the default add handle
            }
        } else if (first == "hogwild" || first == "single" || first == "SPMT") {
            kv = kvstore::KVStore::Get().CreateKVStore<float>();
        } else {
            throw;
        }
        task.set_kvstore(kv);
        husky::LOG_I << GREEN("Set to " + hint + " threads: " + std::to_string(num_train_workers));
    } catch (...) {
        throw base::HuskyException("Unknown hint: " + hint);
    }
    return kv;
}

}  // namespace anonymous
}  // namespace husky
