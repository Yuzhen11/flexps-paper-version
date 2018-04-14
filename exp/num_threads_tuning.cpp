#include "lib/ml_lambda.hpp"

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <numeric>
#include <vector>

#include "datastore/datastore.hpp"
#include "datastore/datastore_utils.hpp"
#include "worker/engine.hpp"

#include "lib/load_data.hpp"
#include "lib/task_utils.hpp"
#include "lib/app_config.hpp"

#include "husky/lib/vector.hpp"
#include "husky/lib/ml/feature_label.hpp"

/*
 * For doing SGD on one process
 * Tuning number of threads for a certain batch size
 *
 * Additional Config:
 * num_load_workers=100
 * tune_batch_size=10,100,1000,10000
 * tune_nums_workers=1,5,10,15,20
 */

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

void sgd_train(const Info& info, datastore::DataStore<LabeledPointHObj<float,float,true>>& data_store, const config::AppConfig& config, long long* computation_time, long long* communication_time) {
    auto start_train = std::chrono::steady_clock::now();
    auto worker = ml::CreateMLWorker<float>(info);
    // Create BatchDataSampler for mini-batch SGD
    datastore::BatchDataSampler<LabeledPointHObj<float, float, true>> batch_data_sampler(data_store, config.batch_size);
    batch_data_sampler.random_start_point();

    for (int iter = 0; iter < config.num_iters; ++iter) {
        // adjust learning rate
        float alpha = config.alpha / (iter / (int)config.learning_rate_coefficient + 1);
        alpha = std::max(1e-5f, alpha);
        // prepare all the indexes in the batch
        std::vector<husky::constants::Key> keys =
            batch_data_sampler.prepare_next_batch();
        std::vector<float> params;
        std::vector<float> delta;
        delta.resize(keys.size(), 0.0);

        auto start_pull = std::chrono::steady_clock::now();
        worker->Pull(keys, &params);                            // issue Pull
        auto end_pull = std::chrono::steady_clock::now();
        if (info.is_leader()) {
            *communication_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_pull - start_pull).count();
        }

        for (auto data : batch_data_sampler.get_data_ptrs()) {  // iterate over the data in the batch
            auto& x = data->x;
            float y = data->y;
            if (y < 0)
                y = 0;
            float pred_y = 0.0;
            int i = 0;
            for (auto field : x) {
                while (keys.at(i) < field.fea)
                    i += 1;
                pred_y += params.at(i) * field.val;
            }
            pred_y = 1. / (1. + exp(-1 * pred_y));
            i = 0;
            for (auto field : x) {
                while (keys.at(i) < field.fea)
                    i += 1;
                delta.at(i) += alpha * field.val * (y - pred_y);
            }
        }

        auto start_push = std::chrono::steady_clock::now();
        worker->Push(keys, delta);  // issue Push
        auto end_push = std::chrono::steady_clock::now();
        if (info.is_leader()) {
            *communication_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_push - start_push).count();
        }
    }

    auto end_train = std::chrono::steady_clock::now();
    if (info.is_leader()) {
        *computation_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_train - start_train).count() - *communication_time;
    }
};

int main(int argc, char** argv) {
    // Set config
    config::InitContext(argc, argv, {"num_load_workers", "tune_nums_workers"});
    auto config = config::SetAppConfigWithContext();
    config.train_epoch = 1;
    config.trainer = "lr";
    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    // different batch sizes and numbers of threads to test
    std::vector<int> nums_workers, batch_sizes;
    auto nums_train_workers = Context::get_param("tune_nums_workers");
    if (nums_train_workers != "") {
        std::vector<std::string> tmp;
        boost::split(tmp, nums_train_workers, boost::is_any_of(","), boost::algorithm::token_compress_on);
        nums_workers.reserve(tmp.size());
        for (auto& t : tmp) {
            nums_workers.push_back(std::stoi(t));
        }
    } else {
        nums_workers.reserve(1);
        nums_workers.push_back(config.num_train_workers);
    }
    auto batch_sizes_str = Context::get_param("tune_batch_sizes");
    assert(batch_sizes_str != "");
    std::vector<std::string> tmp;
    boost::split(tmp, batch_sizes_str, boost::is_any_of(","), boost::algorithm::token_compress_on);
    batch_sizes.reserve(tmp.size());
    for (auto& t : tmp) {
        batch_sizes.push_back(std::stoi(t));
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
    auto load_task = TaskFactory::Get().CreateTask<HuskyTask>(1, num_load_workers);  // 1 epoch
    auto load_task_lambda = [&data_store, config](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, config.num_features, local_id);
    };

    // Submit load_task;
    engine.AddTask(load_task, load_task_lambda); 
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_process_id() == 0)
        husky::LOG_I << YELLOW("Load time: " + std::to_string(load_time) + " ms");

    // Train tasks
    std::vector<MLTask> tasks;
    std::vector<config::AppConfig> task_configs;
    task_configs.reserve(batch_sizes.size() * nums_workers.size());
    for (auto& batch_size : batch_sizes) {
        for (auto& num_workers: nums_workers) {
            if (batch_size % num_workers != 0) {
                husky::LOG_I << CLAY("batch_size is not dividable by num_workers, use approximation.");
            }
            // Create tasks
            auto train_task = TaskFactory::Get().CreateTask<MLTask>();
            config::AppConfig train_config = config;
            train_config.num_train_workers = num_workers;
            train_config.batch_size = (batch_size-1) / num_workers + 1;
            train_config.alpha = config.alpha / num_workers;
            train_task.set_dimensions(train_config.num_params);
            train_task.set_total_epoch(train_config.train_epoch);
            train_task.set_num_workers(train_config.num_train_workers);
            auto hint = config::ExtractHint(train_config);
            int kv = create_kvstore_and_set_hint(hint, train_task, train_config.num_params);
            tasks.push_back(std::move(train_task));
            task_configs.push_back(std::move(train_config));
        }
    }

    // Submit train_task
    long long computation_time = 0, communication_time = 0;
    for (int i = 0; i < tasks.size(); ++ i) {
        engine.AddTask(tasks[i], [&computation_time, &communication_time, &data_store, config = task_configs[i]](const Info& info) {
            sgd_train(info, data_store, config, &computation_time, &communication_time);
        });
        start_time = std::chrono::steady_clock::now();
        engine.Submit();
        end_time = std::chrono::steady_clock::now();
        auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (Context::get_process_id() == 0) {
            husky::LOG_I << "Task, batch_size, num_workers, total time, communication_time, computation_time: "
                << std::to_string(i+1) << "," << (task_configs[i].batch_size*task_configs[i].num_train_workers) << "," << task_configs[i].num_train_workers << ","
                << train_time << "," << communication_time << "," << computation_time;
        }
        computation_time = 0;
        communication_time = 0;
    }

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
