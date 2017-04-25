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
 * num_load_workers=5
 * report_interval=5
 * tune_nums_workers=1,5,10,15,20
 */

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

void sgd_train(const Info& info, datastore::DataStore<LabeledPointHObj<float,float,true>>& data_store, const config::AppConfig& config, int report_interval) {
    auto worker = ml::CreateMLWorker<float>(info);
    // Create BatchDataSampler for mini-batch SGD
    datastore::BatchDataSampler<LabeledPointHObj<float, float, true>> batch_data_sampler(data_store, config.batch_size);
    batch_data_sampler.random_start_point();

    auto start_train = std::chrono::steady_clock::now();
    for (int iter = 0; iter < config.num_iters; ++iter) {
        float alpha = config.alpha / (iter / (int)config.learning_rate_coefficient + 1);
        alpha = std::max(1e-5f, alpha);
        lr::batch_sgd_update_lr(worker, batch_data_sampler, alpha);

        // Report deviations
        if (info.get_cluster_id() == 0 && iter % report_interval == 0) {
            auto current_time = std::chrono::steady_clock::now();
            auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_train).count();
            husky::LOG_I << "Task " << info.get_task_id() << ": Iter, Time: " << iter << "," << train_time;
        }
    }
};

int main(int argc, char** argv) {
    // Set config
    config::InitContext(argc, argv, {"num_load_workers", "report_interval", "tune_nums_workers"});
    auto config = config::SetAppConfigWithContext();
    config.train_epoch = 1;
    config.trainer = "lr";
    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    int report_interval = std::stoi(Context::get_param("report_interval"));
    std::vector<int> nums_workers;
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
    for (auto& num_workers: nums_workers) {
        if (config.batch_size % num_workers != 0) {
            husky::LOG_I << "Warning: batch_size must be dividable by num_workers.";
            continue;
        }
        // Create tasks
        auto train_task = TaskFactory::Get().CreateTask<MLTask>();
        config::AppConfig train_config = config;
        train_config.num_train_workers = num_workers;
        train_config.batch_size = config.batch_size / num_workers;
        train_config.alpha = config.alpha / num_workers;
        train_task.set_dimensions(train_config.num_params);
        train_task.set_total_epoch(train_config.train_epoch);
        train_task.set_num_workers(train_config.num_train_workers);
        auto hint = config::ExtractHint(train_config);
        int kv = create_kvstore_and_set_hint(hint, train_task, train_config.num_params);
        tasks.push_back(std::move(train_task));
        task_configs.push_back(std::move(train_config));
    }

    // Submit train_task
    for (int i = 0; i < tasks.size(); ++ i) {
        engine.AddTask(tasks[i], [&report_interval, &data_store, config = task_configs[i]](const Info& info) {
            sgd_train(info, data_store, config, report_interval);
        });
        start_time = std::chrono::steady_clock::now();
        engine.Submit();
        end_time = std::chrono::steady_clock::now();
        auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (Context::get_process_id() == 0) {
            husky::LOG_I << "Task, num_workers, time: " << std::to_string(i+1) << "," << nums_workers[i] << "," << train_time;
        }
    }

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
