#include "lib/ml_lambda.hpp"

#include <chrono>
#include <vector>

#include "datastore/datastore.hpp"
#include "datastore/datastore_utils.hpp"
#include "worker/engine.hpp"

#include "lib/load_data.hpp"
#include "lib/task_utils.hpp"
#include "lib/app_config.hpp"

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

/*
 * A SGD/mini-batch SGD example
 *
 * Can run in both Single/Hogwild! modes.
 *
 * In each iteration, only Pull the keys needed.
 * For SGD: Pull all the value indexed by one data sample: DataSampler is used.
 * For mini-batch SGD: Pull all the value indexed by a batch of data samples: BatchDataSampler is used.
 *
 * Example:
 *
 * ### Mode
 * kType=PS
 * kConsistency=BSP
 * num_train_workers=4
 * num_load_workers=4
 *
 * input=hdfs:///datasets/classification/a9
 * alpha=0.5
 * num_iters=100
 * num_features=123
 * train_epoch=1
 *
 */
int main(int argc, char** argv) {
    // Set config
    config::InitContext(argc, argv);
    auto config = config::SetAppConfigWithContext();
    if (Context::get_worker_info().get_process_id() == 0)
        config:: ShowConfig(config);
    auto hint = config::ExtractHint(config);

    auto& engine = Engine::Get();
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());
    // Create the DataStore
    datastore::DataStore<LabeledPointHObj<float, float, true>> data_store(
        Context::get_worker_info().get_num_local_workers());

    // Load task
    auto load_task = TaskFactory::Get().CreateTask<HuskyTask>(1, config.num_load_workers);  // 1 epoch
    auto load_task_lambda = [&data_store, config](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, config.num_features, local_id);
    };

    // Train task
    std::vector<MLTask> tasks;
    std::vector<config::AppConfig> configs;
    for (int i = 0; i < 10; ++ i) {
        auto train_task = TaskFactory::Get().CreateTask<MLTask>();
        train_task.set_dimensions(config.num_params);
        train_task.set_total_epoch(config.train_epoch);  // set epoch number
        train_task.set_num_workers(config.num_train_workers);
        int kv1 = create_kvstore_and_set_hint(hint, train_task, config.num_params);  // Create KVStore and Set hint
        assert(kv1 != -1);
        tasks.push_back(std::move(train_task));

        // Using different learning rate
        config.alpha = 0.1 + i*0.1;
        husky::LOG_I << RED("Warning: Setting alpha to "+std::to_string(config.alpha));
        configs.push_back(config);
    }

    // Submit load_task;
    engine.AddTask(load_task, load_task_lambda); 
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    husky::LOG_I << YELLOW("Load time: " + std::to_string(load_time) + " ms");

    // Submit train_task
    for (int i = 0; i < tasks.size(); ++ i) {
        engine.AddTask(tasks[i], [&data_store, config = configs[i]](const Info& info) {
            lambda::train(data_store, config, info);
        });
    }
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    husky::LOG_I << YELLOW("train time: " + std::to_string(train_time) + " ms");

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
