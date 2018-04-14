#pragma once

#include <vector>
#include <chrono>
#include <kvstore/kvstore.hpp>

#include "husky/core/context.hpp"
#include "core/color.hpp"

namespace ml {
namespace model {
namespace {

/*
 * LoadIntegralFromKV is to load all the parameters and store in std::vector
 */
template<typename Val>
void LoadIntegralFromKV(int local_id, int model_id, int num_params, std::vector<Val>* params) {
    husky::LOG_I << PURPLE("[LoadIntegralFromKV] model_id: " + std::to_string(model_id) + " local_id: " +
                        std::to_string(local_id) + " model_size: " + std::to_string(num_params));
    auto start_time = std::chrono::steady_clock::now();
    auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
    // TODO change the keys to chunk based or add another PullAll APIs to kvstore
    std::vector<husky::constants::Key> keys(num_params);
    for (size_t i = 0; i < keys.size(); ++i)
        keys[i] = i;
    int ts = kvworker->Pull(model_id, keys, params);
    kvworker->Wait(model_id, ts);
    auto end_time = std::chrono::steady_clock::now();
    husky::LOG_I << PURPLE("[LoadIntegralFromKV] Load time: "
                 + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count())
                 + " ms");
}

/*
 * Load the params from ModelTransferManager
 */
template<typename Val>
void LoadIntegralFromStore(int local_id, int model_id, std::vector<Val>* params) {
    husky::LOG_I << PURPLE("[LoadIntegralFromStore] model_id: " + std::to_string(model_id) + " local_id: " + 
                        std::to_string(local_id));
    // Receive from mailbox
    auto* mailbox = husky::Context::get_mailbox(local_id);
    if (mailbox->poll(0,0)) {
        auto bin = mailbox->recv(0,0);
        bin >> *params;
    }
}

/*
 * LoadChunks is to load part of the chunks
 */
template<typename Val>
void LoadChunks(int local_id, int model_id,
        const std::vector<size_t>& keys, std::vector<std::vector<Val>*>* chunks) {
    auto start_time = std::chrono::steady_clock::now();
    auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
    int ts = kvworker->PullChunks(model_id, keys, *chunks, false);
    kvworker->Wait(model_id, ts);
    auto end_time = std::chrono::steady_clock::now();
    husky::LOG_I << PURPLE("[LoadChunks] Load time: "
                 + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count())
                 + " ms");
}

/*
 * LoadAllChunks is to load all the parameters and stored in chunks
 * Make sure that the chunks are ready
 * TODO: This part can actually be parallelized
 */
template<typename Val>
void LoadAllChunksFromKV(int local_id, int model_id, std::vector<std::vector<Val>>* chunks) {
    husky::LOG_I << PURPLE("[LoadAllChunksFromKV] model_id:" + std::to_string(model_id) + " local_id:" +
                        std::to_string(local_id));
    std::vector<size_t> keys;
    keys.reserve(chunks->size());
    std::vector<std::vector<Val>*> params;
    params.reserve(chunks->size());
    for (size_t i = 0; i < chunks->size(); ++ i) {
        keys.push_back(i);
        params.push_back(&(*chunks)[i]);
    }
    LoadChunks(local_id, model_id, keys, &params);
}

template<typename Val>
void LoadAllChunksFromStore(int local_id, int model_id, std::vector<std::vector<Val>>* chunks) {
    husky::LOG_I << PURPLE("[LoadAllChunksFromStore] model_id: " + std::to_string(model_id) + " local_id: " + 
                        std::to_string(local_id));
    // Receive from mailbox
    auto* mailbox = husky::Context::get_mailbox(local_id);
    if (mailbox->poll(0,0)) {
        auto bin = mailbox->recv(0,0);
        bin >> *chunks;
    }
}

}  // namespace anonymous
}  // namespace model
}  // namespace ml
