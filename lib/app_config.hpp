#pragma once

#include <cassert>
#include <map>
#include <string>
#include <sstream>
#include "core/constants.hpp"
#include "core/color.hpp"
#include "husky/base/log.hpp"
#include "husky/core/context.hpp"
#include "husky/core/job_runner.hpp"

namespace husky {
namespace config {

/*
 * Only store the ml-related configure parameters
 */
struct AppConfig {
    int train_epoch = 1;
    float alpha = 0.0;
    int num_iters = -1;
    int num_features = -1;  // to be deleted
    int num_params = -1;
    std::string kType;
    std::string kConsistency;
    int num_train_workers = -1;
    std::string trainer;
    bool use_chunk = false;
    bool use_direct_model_transfer = false;
    int staleness = 1;
    std::string ps_worker_type;
    float learning_rate_coefficient = 0.0f;
    std::string learning_rate_update;
};

namespace {

void ShowConfig(const AppConfig& config) {
    std::stringstream ss;
    ss << "Showing config: ";
    ss << "\ntrain_epoch: " << config.train_epoch;
    ss << "\nalpha: " << config.alpha;
    ss << "\nnum_iters: " << config.num_iters;
    ss << "\nnum_features: " << config.num_features;
    ss << "\nnum_params: " << config.num_params;
    ss << "\nkType: " << config.kType;
    ss << "\nkConsistency: " << config.kConsistency;
    ss << "\nnum_train_workers: " << config.num_train_workers;
    ss << "\ntrainer: " << config.trainer;
    ss << "\nuse_chunk: " << config.use_chunk;
    ss << "\nuse_direct_model_transfer: " << config.use_direct_model_transfer;
    ss << "\nstaleness: " << config.staleness;
    husky::LOG_I << RED(ss.str());
}

void InitContext(int argc, char** argv, const std::vector<std::string>& additional_args = {}) {
    std::vector<std::string> default_init_args = 
    {"worker_port", "cluster_manager_host", "cluster_manager_port", "hdfs_namenode",
     "hdfs_namenode_port", "input", "num_features", "alpha", "num_iters", "train_epoch",
     "kType", "kConsistency", "num_train_workers", "trainer", 
     "use_chunk", "use_direct_model_transfer", "staleness", "ps_worker_type"};

    std::vector<std::string> init_args = default_init_args;
    // Add additional args
    init_args.insert(init_args.end(), additional_args.begin(), additional_args.end());

    bool rt =
        init_with_args(argc, argv, init_args);
    if (!rt)
        assert(0);
}

AppConfig SetAppConfigWithContext() {
    AppConfig config;
    config.train_epoch = std::stoi(Context::get_param("train_epoch"));
    config.alpha = std::stof(Context::get_param("alpha"));
    config.num_iters = std::stoi(Context::get_param("num_iters"));
    config.num_features = std::stoi(Context::get_param("num_features"));
    config.num_params = config.num_features + 1;  // +1 for starting from 1, but not for intercept
    config.kType = Context::get_param("kType");
    config.kConsistency = Context::get_param("kConsistency");
    config.num_train_workers = std::stoi(Context::get_param("num_train_workers"));
    config.trainer = Context::get_param("trainer");
    config.use_chunk = Context::get_param("use_chunk") == "on" ? true : false;
    config.ps_worker_type = Context::get_param("ps_worker_type");
    config.use_direct_model_transfer = Context::get_param("use_direct_model_transfer")  == "on" ? true : false;
    config.staleness = std::stoi(Context::get_param("staleness"));
    if (Context::get_param("learning_rate_coefficient") != "") {
        config.learning_rate_coefficient = std::stof(Context::get_param("learning_rate_coefficient"));
    }
    config.learning_rate_update = Context::get_param("learning_rate_update");

    const std::vector<std::string> trainers_set({"lr", "svm"});
    assert(std::find(trainers_set.begin(), trainers_set.end(), config.trainer) != trainers_set.end());
    husky::LOG_I << CLAY("Trainer: "+config.trainer);
    husky::LOG_I << CLAY("use_chunk: "+std::to_string(config.use_chunk));
    husky::LOG_I << CLAY("use_direct_model_transfer: "+std::to_string(config.use_direct_model_transfer));
    return config;
}

std::map<std::string, std::string> ExtractHint(const AppConfig& config) {
    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kType, config.kType},
        {husky::constants::kConsistency, config.kConsistency},
        {husky::constants::kNumWorkers, std::to_string(config.num_train_workers)},
        {husky::constants::kStaleness, std::to_string(config.staleness)},  // default staleness
        {husky::constants::kStorageType, husky::constants::kVectorStorage}  // Use vector storage
    };
    
    if (config.kType == husky::constants::kPS && config.kConsistency == husky::constants::kSSP) {
        const std::vector<std::string> ps_worker_types{"PSWorker", "PSMapNoneWorker", "PSChunkNoneWorker", "PSNoneChunkWorker", "PSMapChunkWorker", "PSChunkChunkWorker"};
        assert(std::find(ps_worker_types.begin(), ps_worker_types.end(), config.ps_worker_type) != ps_worker_types.end());
        hint.insert({husky::constants::kWorkerType, config.ps_worker_type});
    }
    if (config.use_chunk) {
        hint.insert({husky::constants::kParamType, husky::constants::kChunkType});
    }
    if (config.use_direct_model_transfer) {
        hint.insert({husky::constants::kEnableDirectModelTransfer, "on"});
    }
    if (config.use_chunk && config.use_direct_model_transfer) {
        assert(false);
    }
    return hint;
}


}  // namespace anonymous
}  // namespace config 
}  // namespace husky
