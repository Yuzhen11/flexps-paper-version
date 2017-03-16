#pragma once

#include <vector>
#include <chrono>
#include "kvstore/kvstore.hpp"
#include "worker/model_transfer_store.hpp"
#include "core/color.hpp"

namespace ml {
namespace model {
namespace {


template<typename Val>
void DumpAllIntegral(int local_id, int model_id, int num_params, 
        const std::vector<Val>& params) {
    husky::LOG_I << PURPLE("[DumpAllIntegral] Dumping model_id: " + std::to_string(model_id) + " local_id: " +
                        std::to_string(local_id) + " model_size: " + std::to_string(num_params));
    auto start_time = std::chrono::steady_clock::now();
    auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
    std::vector<husky::constants::Key> keys(num_params);
    for (size_t i = 0; i < keys.size(); ++i)
        keys[i] = i;
    int ts = kvworker->Push(model_id, keys, params);
    kvworker->Wait(model_id, ts);
    auto end_time = std::chrono::steady_clock::now();
    husky::LOG_I << PURPLE("[DumpAllIntegral] Dump time: "
                 + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count())
                 + " ms");
}

/*
 * Dump the params into model_transfer_store
 */
template<typename Val>
void DumpIntegralToStore(int model_id, std::vector<Val>&& params) {
    husky::LOG_I << PURPLE("[DumpIntegralToStore] Dumping model_id: " + std::to_string(model_id)
            + "model_size: "+std::to_string(params.size()));
    auto& store = husky::ModelTransferStore::Get();
    husky::base::BinStream bin;
    bin << params;
    store.Add(model_id, std::move(bin));
}

template<typename Val>
void DumpChunks(int local_id, int model_id,
        const std::vector<size_t>& keys, const std::vector<std::vector<Val>*>& chunks) {
    husky::LOG_I << PURPLE("[DumpChunks] Dumping model_id:" + std::to_string(model_id) + " local_id: " +
                        std::to_string(local_id) + " chunk_num: " + std::to_string(keys.size()));
    auto start_time = std::chrono::steady_clock::now();
    auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
    int ts = kvworker->PushChunks(model_id, keys, chunks, false);
    kvworker->Wait(model_id, ts);
    auto end_time = std::chrono::steady_clock::now();
    husky::LOG_I << PURPLE("[DumpChunks] Dump time: "
                 + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count())
                 + " ms");
}

template<typename Val>
void DumpAllChunks(int local_id, int model_id, const std::vector<std::vector<Val>>& chunks) {
    std::vector<size_t> keys;
    keys.reserve(chunks.size());
    std::vector<std::vector<Val>*> params;
    params.reserve(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++ i) {
        keys.push_back(i);
        params.push_back(const_cast<std::vector<Val>*>(&chunks[i]));
    }
    DumpChunks(local_id, model_id, keys, params);
}

}  // namespace anonymous
}  // namespace model
}  // namespace ml
