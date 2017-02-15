#pragma once

#include <vector>
#include <chrono>
#include <kvstore/kvstore.hpp>

namespace ml {
namespace model {
namespace {

/*
 * LoadAllIntegral is to load all the parameters and store in std::vector
 */
void LoadAllIntegral(int local_id, int model_id, int num_params, std::vector<float>* params) {
    husky::LOG_I << "[LoadAllIntegral] Loading model_id:" + std::to_string(model_id) + " local_id:" +
                        std::to_string(local_id) + " model_size: " + std::to_string(num_params);
    auto start_time = std::chrono::steady_clock::now();
    auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
    // TODO change the keys to chunk based or add another PullAll APIs to kvstore
    std::vector<husky::constants::Key> keys(num_params);
    for (size_t i = 0; i < keys.size(); ++i)
        keys[i] = i;
    int ts = kvworker->Pull(model_id, keys, params);
    kvworker->Wait(model_id, ts);
    auto end_time = std::chrono::steady_clock::now();
    husky::LOG_I << "[LoadAllIntegral] Load time: "
                 << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
                 << " ms";
}

/*
 * LoadChunks is to load part of the chunks
 */
void LoadChunks(int local_id, int model_id,
        const std::vector<size_t>& keys, std::vector<std::vector<float>*>* chunks) {
    husky::LOG_I << "[LoadChunks] loading model_id:" + std::to_string(model_id) + " local_id:" +
                        std::to_string(local_id) + " chunk_num: " + std::to_string(keys.size());
    auto start_time = std::chrono::steady_clock::now();
    auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
    int ts = kvworker->PullChunks(model_id, keys, *chunks, false);
    kvworker->Wait(model_id, ts);
    auto end_time = std::chrono::steady_clock::now();
    husky::LOG_I << "[LoadChunks] Load time: "
                 << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
                 << " ms";
}

/*
 * LoadAllChunks is to load all the parameters and stored in chunks
 * Make sure that the chunks are ready
 * TODO: This part can actually be parallelized
 */
void LoadAllChunks(int local_id, int model_id, std::vector<std::vector<float>>* chunks) {
    std::vector<size_t> keys;
    keys.reserve(chunks->size());
    std::vector<std::vector<float>*> params;
    params.reserve(chunks->size());
    for (size_t i = 0; i < chunks->size(); ++ i) {
        keys.push_back(i);
        params.push_back(&(*chunks)[i]);
    }
    LoadChunks(local_id, model_id, keys, &params);
}

}  // namespace anonymous
}  // namespace model
}  // namespace ml
