#pragma once

#include <vector>
#include <chrono>
#include <kvstore/kvstore.hpp>

namespace ml {
namespace spmt {
namespace {

void DumpAllIntegral(int local_id, int model_id, int num_params, 
        const std::vector<float>& params) {
    husky::LOG_I << "[DumpAllIntegral] Dumping model_id:" + std::to_string(model_id) + " local_id:" +
                        std::to_string(local_id) + " model_size: " + std::to_string(num_params);
    auto start_time = std::chrono::steady_clock::now();
    auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
    std::vector<husky::constants::Key> keys(num_params);
    for (size_t i = 0; i < keys.size(); ++i)
        keys[i] = i;
    int ts = kvworker->Push(model_id, keys, params);
    kvworker->Wait(model_id, ts);
    auto end_time = std::chrono::steady_clock::now();
    husky::LOG_I << "[DumpAllIntegral] Dump time: "
                 << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
                 << " ms";
}

void DumpChunks(int local_id, int model_id,
        const std::vector<size_t>& keys, const std::vector<std::vector<float>*>& chunks) {
    husky::LOG_I << "[DumpChunks] Dumping model_id:" + std::to_string(model_id) + " local_id:" +
                        std::to_string(local_id) + " chunk_num: " + std::to_string(keys.size());
    auto start_time = std::chrono::steady_clock::now();
    auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
    int ts = kvworker->PushChunks(model_id, keys, chunks, false);
    kvworker->Wait(model_id, ts);
    auto end_time = std::chrono::steady_clock::now();
    husky::LOG_I << "[DumpChunks] Dump time: "
                 << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
                 << " ms";
}

void DumpAllChunks(int local_id, int model_id, const std::vector<std::vector<float>>& chunks) {
    std::vector<size_t> keys;
    keys.reserve(chunks.size());
    std::vector<std::vector<float>*> params;
    params.reserve(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++ i) {
        keys.push_back(i);
        params.push_back(const_cast<std::vector<float>*>(&chunks[i]));
    }
    DumpChunks(local_id, model_id, keys, params);
}

}  // namespace anonymous
}  // namespace spmt
}  // namespace ml
