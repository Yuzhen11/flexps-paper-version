#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <numeric>
#include <vector>

#include "datastore/datastore.hpp"
#include "datastore/datastore_utils.hpp"
#include "worker/engine.hpp"

#include "lib/app_config.hpp"
#include "lib/load_data.hpp"
#include "lib/lr_test_utils.hpp"
#include "lib/task_utils.hpp"

#include "husky/lib/vector.hpp"
#include "husky/lib/ml/feature_label.hpp"

/*
 * For doing SGD on one process
 * Tuning: batch_size, learning rate, and learning rate update coefficient
 *
 * Additional Config:
 * num_load_workers=5
 * report_interval=5
 * model_input=hdfs:///ml/syn_1000_model
 * tune_learning_rate=0.01,0.05,0.1,0.2,0.5,1
 * tune_lr_coeff=10,100,1000
 * tune_batch_size=10,20,30,40,50,100,200,300,400,500,1000,2000,3000
 */

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

int main(int argc, char** argv) {
    // Set config
    config::InitContext(argc, argv, {"num_load_workers", "model_input", "report_interval", "tune_learning_rate", "tune_lr_coeff", "tune_batch_size"});
    assert(Context::get_num_processes() == 1);
    auto config = config::SetAppConfigWithContext();
    config.train_epoch = 1;
    config.trainer = "lr";
    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    auto model_file = Context::get_param("model_input");
    int report_interval = std::stoi(Context::get_param("report_interval"));
    std::vector<float> learning_rates;
    std::vector<int> lr_coeffs;
    std::vector<int> batch_sizes;
    auto tune_learning_rate = Context::get_param("tune_learning_rate");
    auto tune_lr_coeff = Context::get_param("tune_lr_coeff");
    auto tune_batch_size = Context::get_param("tune_batch_size");
    if (tune_learning_rate != "") {
        std::vector<std::string> tmp;
        boost::split(tmp, tune_learning_rate, boost::is_any_of(","), boost::algorithm::token_compress_on);
        learning_rates.reserve(tmp.size());
        for (auto& t : tmp) {
            learning_rates.push_back(std::stof(t));
        }
    } else {
        learning_rates.reserve(1);
        learning_rates.push_back(config.alpha);
    }
    if (tune_lr_coeff != "") {
        std::vector<std::string> tmp;
        boost::split(tmp, tune_lr_coeff, boost::is_any_of(","), boost::algorithm::token_compress_on);
        lr_coeffs.reserve(tmp.size());
        for (auto& t : tmp) {
            lr_coeffs.push_back(std::stoi(t));
        }
    } else {
        lr_coeffs.reserve(1);
        lr_coeffs.push_back(config.learning_rate_coefficient);
    }
    if (tune_batch_size != "") {
        std::vector<std::string> tmp;
        boost::split(tmp, tune_batch_size, boost::is_any_of(","), boost::algorithm::token_compress_on);
        batch_sizes.reserve(tmp.size());
        for (auto& t : tmp) {
            batch_sizes.push_back(std::stoi(t));
        }
    } else {
        batch_sizes.reserve(1);
        batch_sizes.push_back(config.batch_size);
    }
            
    // Start engine
    auto& engine = Engine::Get();
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());
    // Create the DataStore
    datastore::DataStore<LabeledPointHObj<float, float, true>> data_store(
        Context::get_worker_info().get_num_local_workers());

    // Load task
    int kv_true = -1;
    std::vector<float> benchmark_model;
    if (model_file != "") {
        kv_true = kvstore::KVStore::Get().CreateKVStore<float>();
        assert(kv_true != -1);
    }
    auto load_task = TaskFactory::Get().CreateTask<HuskyTask>(1, num_load_workers);  // 1 epoch
    auto load_task_lambda = [&data_store, &benchmark_model, &model_file, &kv_true, config](const Info& info) {
        auto local_id = info.get_local_id();

        // Load benchmark model if provided
        if (model_file != "" && info.get_cluster_id() == 0) {
            lr::load_benchmark_model(benchmark_model, model_file, config.num_params);
            std::vector<husky::constants::Key> keys(config.num_params);
            for (int i = 0; i < config.num_params; ++i) keys[i] = i;
            auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
            kvworker->Wait(kv_true, kvworker->Push(kv_true, keys, benchmark_model));
        }

        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, config.num_features, local_id);
    };

    // Train tasks
    std::vector<Task> tasks;
    std::vector<config::AppConfig> task_configs;
    std::vector<int> kvs;
    for (auto& batch_size : batch_sizes) {
        for (auto& learning_rate : learning_rates) {
            for (auto& lr_coeff : lr_coeffs) {
                // Create tasks
                auto train_task = TaskFactory::Get().CreateTask<Task>();
                config::AppConfig train_config = config;
                train_config.batch_size = batch_size;
                train_config.learning_rate_coefficient = lr_coeff;
                train_config.alpha = learning_rate;
                train_task.set_dimensions(train_config.num_params);
                train_task.set_total_epoch(train_config.train_epoch);
                train_task.set_num_workers(train_config.num_train_workers);
                auto hint = config::ExtractHint(train_config);
                int kv = create_kvstore_and_set_hint(hint, train_task, train_config.num_params);
                kvs.push_back(kv);
                tasks.push_back(std::move(train_task));
                task_configs.push_back(train_config);
            }
        }
    }

    // Submit load_task;
    engine.AddTask(load_task, load_task_lambda); 
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_process_id() == 0)
        husky::LOG_I << YELLOW("Load time: " + std::to_string(load_time) + " ms");

    // Submit train_task
    for (int i = 0; i < tasks.size(); ++ i) {
        // TODO mutable is not safe when the lambda is used more than once
        engine.AddTask(tasks[i], [&benchmark_model, kv_true, &report_interval, &data_store, config = task_configs[i]](const Info& info) mutable {
            assert(config.batch_size % config.num_train_workers == 0);
            if (info.get_cluster_id() == 0 && kv_true != -1 && benchmark_model.empty()) {
                std::vector<husky::constants::Key> keys(config.num_params);
                for (int i = 0; i < config.num_params; ++i) keys[i] = i;
                auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
                kvworker->Wait(kv_true, kvworker->Pull(kv_true, keys, &benchmark_model));
            }
            config.batch_size = config.batch_size / config.num_train_workers;
            config.alpha = config.alpha / config.num_train_workers;
            lr::sgd_train(info, data_store, config, benchmark_model, report_interval);
        });
        if (Context::get_process_id() == 0) {
            husky::LOG_I << RED("task_config " + std::to_string(i));
            config::ShowConfig(task_configs[i]);
        }
    }
    engine.Submit();

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
