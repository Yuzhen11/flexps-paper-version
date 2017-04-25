#include "lib/ml_lambda.hpp"

#include <chrono>
#include <vector>

#include "datastore/datastore.hpp"
#include "datastore/datastore_utils.hpp"
#include "worker/engine.hpp"

#include "lib/load_data.hpp"
#include "lib/task_utils.hpp"
#include "lib/app_config.hpp"

#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string.hpp"

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

/*
 * Train mulitple models (different types of models) to test different scheduler
 *
 * conf:
 * # choose one of the followings:
 * task_scheduler_type=sequential
 * task_scheduler_type=greedy
 * task_scheduler_type=priority
 *
 * # for MultiTasksExp
 * num_workers=5,100,100,40,40,280
 * num_stages=6,4,6,5,5,3
 * time=300,700,500,600,700,300
 * type=SPMT,PS,PS,PS,PS,PS
 */
int main(int argc, char** argv) {
    // Set config
    config::InitContext(argc, argv, {"num_load_workers", "num_workers", "num_stages", "time", "type"});
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
    // int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    // auto load_task = TaskFactory::Get().CreateTask<HuskyTask>(1, num_load_workers);  // 1 epoch
    // auto load_task_lambda = [&data_store, config](const Info& info) {
    //     auto local_id = info.get_local_id();
    //     load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, config.num_features, local_id);
    // };

    // Train task
    std::vector<MLTask> tasks;
    std::vector<config::AppConfig> task_configs;
    std::vector<int> kvs;
    std::vector<std::function<void()>> funcs;
    {
        std::string num_workers_str = Context::get_param("num_workers");
        std::string num_stages_str = Context::get_param("num_stages");
        std::string time_str = Context::get_param("time");
        std::string type_str = Context::get_param("type");

        std::vector<std::string> num_workers_splitted;
        std::vector<std::string> num_stages_splitted;
        std::vector<std::string> time_splitted;
        std::vector<std::string> type_splitted;
        boost::split(num_workers_splitted, num_workers_str, boost::is_any_of(", "), boost::token_compress_on);
        boost::split(num_stages_splitted, num_stages_str, boost::is_any_of(", "), boost::token_compress_on);
        boost::split(time_splitted, time_str, boost::is_any_of(", "), boost::token_compress_on);
        boost::split(type_splitted, type_str, boost::is_any_of(", "), boost::token_compress_on);

        assert(num_workers_splitted.size() == num_stages_splitted.size() && num_stages_splitted.size() == time_splitted.size() && time_splitted.size() == type_splitted.size());
        for (int i = 0; i < num_workers_splitted.size(); ++ i) {
            auto train_task = TaskFactory::Get().CreateTask<MLTask>();
            config::AppConfig train_config = config;
            assert(type_splitted[i] == "PS" || type_splitted[i] == "SPMT");
            train_config.kType = type_splitted[i];  // type
            train_config.kConsistency = "SSP";
            train_config.staleness = 1;
            train_config.ps_worker_type = "PSWorker";
            train_config.num_train_workers = std::stoi(num_workers_splitted[i]);  // workers
            train_config.train_epoch = std::stoi(num_stages_splitted[i]);  // stages
            train_config.num_iters = 100;
            train_task.set_dimensions(train_config.num_params);
            train_task.set_total_epoch(train_config.train_epoch);
            train_task.set_num_workers(train_config.num_train_workers);
            auto hint = config::ExtractHint(train_config);
            int kv = create_kvstore_and_set_hint(hint, train_task, train_config.num_params);
            kvs.push_back(kv);
            tasks.push_back(std::move(train_task));
            task_configs.push_back(train_config);
            funcs.push_back([id = train_task.get_id(), sleep_time = time_splitted[i]]() {  // time
                husky::LOG_I << "Task " << id << " is running";
                std::this_thread::sleep_for(std::chrono::milliseconds(std::stoi(sleep_time)));  // time
            });
        }
    }
    /*
    {
        // task 1
        auto train_task = TaskFactory::Get().CreateTask<MLTask>();
        config::AppConfig train_config = config;
        train_config.kType = "PS";
        train_config.kConsistency = "SSP";
        train_config.staleness = 1;
        train_config.ps_worker_type = "PSWorker";
        train_config.num_train_workers = 10;
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
        funcs.push_back([id = train_task.get_id()]() {
            husky::LOG_I << "Task " << id << " is running";
        });
    }
    {
        // task 2
        auto train_task = TaskFactory::Get().CreateTask<MLTask>();
        config::AppConfig train_config = config;
        train_config.kType = "Hogwild";
        train_config.num_train_workers = 4;
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
        funcs.push_back([id = train_task.get_id()]() {
            husky::LOG_I << "Task " << id << " is running";
        });
    }
    */
    // add 5 ps jobs
    /*
    for (int i = 0; i < 5; ++ i) {
        auto train_task = TaskFactory::Get().CreateTask<MLTask>();
        config::AppConfig train_config = config;
        train_config.kType = "PS";
        train_config.kConsistency = "SSP";
        train_config.staleness = 1;
        train_config.ps_worker_type = "PSWorker";
        train_config.num_train_workers = 30;
        train_config.train_epoch = 5;
        train_config.num_iters = 100;
        train_task.set_dimensions(train_config.num_params);
        train_task.set_total_epoch(train_config.train_epoch);
        train_task.set_num_workers(train_config.num_train_workers);
        auto hint = config::ExtractHint(train_config);
        int kv = create_kvstore_and_set_hint(hint, train_task, train_config.num_params);
        kvs.push_back(kv);
        tasks.push_back(std::move(train_task));
        task_configs.push_back(train_config);
        funcs.push_back([id = train_task.get_id()]() {
            husky::LOG_I << "Task " << id << " is running";
        });
    }
    // add 2 spmt jobs
    for (int i = 0; i < 2; ++ i) {
        auto train_task = TaskFactory::Get().CreateTask<MLTask>();
        config::AppConfig train_config = config;
        train_config.kType = "SPMT";
        train_config.num_train_workers = 5;
        train_config.train_epoch = 10;
        train_config.num_iters = 100;
        train_task.set_dimensions(train_config.num_params);
        train_task.set_total_epoch(train_config.train_epoch);
        train_task.set_num_workers(train_config.num_train_workers);
        auto hint = config::ExtractHint(train_config);
        int kv = create_kvstore_and_set_hint(hint, train_task, train_config.num_params);
        kvs.push_back(kv);
        tasks.push_back(std::move(train_task));
        task_configs.push_back(train_config);
        funcs.push_back([id = train_task.get_id()]() {
            husky::LOG_I << "Task " << id << " is running";
        });
    }
    for (int i = 0; i < 3; ++ i) {
        auto train_task = TaskFactory::Get().CreateTask<MLTask>();
        config::AppConfig train_config = config;
        train_config.kType = "PS";
        train_config.kConsistency = "SSP";
        train_config.staleness = 1;
        train_config.ps_worker_type = "PSWorker";
        train_config.num_train_workers = 300;
        train_config.train_epoch = 3;
        train_config.num_iters = 50;
        train_task.set_dimensions(train_config.num_params);
        train_task.set_total_epoch(train_config.train_epoch);
        train_task.set_num_workers(train_config.num_train_workers);
        auto hint = config::ExtractHint(train_config);
        int kv = create_kvstore_and_set_hint(hint, train_task, train_config.num_params);
        kvs.push_back(kv);
        tasks.push_back(std::move(train_task));
        task_configs.push_back(train_config);
        funcs.push_back([id = train_task.get_id()]() {
            husky::LOG_I << "Task " << id << " is running";
        });
    }
    */

    // Submit load_task;
    // engine.AddTask(load_task, load_task_lambda); 
    auto start_time = std::chrono::steady_clock::now();
    // engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_process_id() == 0)
        husky::LOG_I << YELLOW("Load time: " + std::to_string(load_time) + " ms");

    // Submit train_task
    for (int i = 0; i < tasks.size(); ++ i) {
        engine.AddTask(tasks[i], [i, &funcs, &data_store, config = task_configs[i]](const Info& info) {
            // lambda::train(data_store, config, info);
            funcs[i]();
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
