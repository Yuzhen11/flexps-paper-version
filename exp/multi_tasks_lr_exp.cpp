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
 * Train multiple lr models
 */
int main(int argc, char** argv) {
    // Set config
    config::InitContext(argc, argv, {"num_load_workers"});
    auto config = config::SetAppConfigWithContext();
    // if (Context::get_worker_info().get_process_id() == 0)
    //     config:: ShowConfig(config);
    auto hint = config::ExtractHint(config);

    auto& engine = Engine::Get();
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());
    // Create the DataStore
    datastore::DataStore<LabeledPointHObj<float, float, true>> data_store(
        Context::get_worker_info().get_num_local_workers());

    // Load task
    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    auto load_task = TaskFactory::Get().CreateTask<HuskyTask>(1, num_load_workers);  // 1 epoch
    auto load_task_lambda = [&data_store, config](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, config.num_features, local_id);
    };

    // Train task
    std::vector<MLTask> tasks;
    std::vector<config::AppConfig> task_configs;
    std::vector<int> kvs;
    // add 10 ps jobs
    for (int i = 0; i < 10; ++ i) {
        auto train_task = TaskFactory::Get().CreateTask<MLTask>();
        config::AppConfig train_config = config;
        train_config.kType = "PS";
        train_config.kConsistency = "SSP";
        train_config.staleness = 2;
        train_config.ps_worker_type = "PSWorker";
        train_config.num_train_workers = 30;
        train_config.train_epoch = 1;
        train_config.num_iters = 100;
        train_task.set_dimensions(train_config.num_params);
        train_task.set_total_epoch(train_config.train_epoch);
        train_task.set_num_workers(train_config.num_train_workers);
        auto hint = config::ExtractHint(train_config);
        int kv = create_kvstore_and_set_hint(hint, train_task, train_config.num_params);
        kvs.push_back(kv);
        tasks.push_back(std::move(train_task));
        task_configs.push_back(train_config);
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
        engine.AddTask(tasks[i], [i, &data_store, config = task_configs[i]](const Info& info) {
            husky::LOG_I << "Task: " << info.get_task_id() << " is running";
            lambda::train(data_store, config, info);
            // std::this_thread::sleep_for(std::chrono::milliseconds(500));
        });
        if (Context::get_process_id() == 0) {
            husky::LOG_I << RED("task_config " + std::to_string(i));
            config::ShowConfig(task_configs[i]);
        }
    }
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_process_id() == 0)
        husky::LOG_I << YELLOW("train time: " + std::to_string(train_time) + " ms");

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
