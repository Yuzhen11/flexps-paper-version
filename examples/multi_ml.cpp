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

// Perform lr_test
bool lr_test(const LabeledPointHObj<float, float, true>& data, const std::vector<float>& model) {
    auto& x = data.x;
    float y = data.y;
    if (y < 0)
        y = 0;
    float pred_y = 0.0;
    for (auto field : x) {
        pred_y += model[field.fea] * field.val;
    }
    pred_y = 1. / (1. + exp(-pred_y));
    pred_y = (pred_y > 0.5) ? 1 : 0;
    return int(pred_y) == int(y) ? true : false;
}

void test_lambda(datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store,
        std::vector<int> kvs,
        int num_params,
        const Info& info) {
    // find pos for DataLoadBalance
    auto local_tids = info.get_local_tids();
    int pos = -1;
    for (int i = 0; i < local_tids.size(); ++ i) {
        if (local_tids[i] == info.get_global_id()) {
            pos = i;
            break;
        }
    }
    assert(pos != -1);
    // use DataLoadBalance to work on a partition of local data
    datastore::DataLoadBalance<LabeledPointHObj<float, float, true>> data_load_balance(data_store, local_tids.size(), pos);

    // Pull models from kvstore
    std::vector<std::vector<float>> models(kvs.size());
    auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
    std::vector<husky::constants::Key> all_keys(num_params);
    std::iota(all_keys.begin(), all_keys.end(), 0);
    for (int i = 0; i < kvs.size(); ++ i) {
        kvworker->Wait(kvs[i], kvworker->Pull(kvs[i], all_keys, &models[i], false, true, false));
    }
    husky::LOG_I << "Model prepared";
    int local_data_count = 0;
    int local_data_correct_count = 0;
    while (data_load_balance.has_next()) {
        auto& data = data_load_balance.next();
        int correct_count = 0;
        for (auto& model : models) {
            bool correct = lr_test(data, model);
            if (correct) correct_count += 1;
        }
        if (correct_count > models.size()/2) {  // poll
            local_data_correct_count += 1;
        }
        local_data_count += 1;
    }
    if (local_data_count > 0)
        husky::LOG_I << "Local test result: (" << local_data_correct_count << "/" << local_data_count << ")"
            << " Local ensemble accuracy: " << local_data_correct_count*1.0/local_data_count;
    else
        husky::LOG_I << "Empty local data";

    // Aggregate the result
    int dst = info.get_tid(0); // get the tids of 0 cluster id
    auto* mailbox = Context::get_mailbox(info.get_local_id());
    husky::base::BinStream bin;
    bin << local_data_correct_count << local_data_count;
    mailbox->send(dst, 0, 0, bin);
    mailbox->send_complete(0, 0, 
            info.get_worker_info().get_local_tids(), info.get_worker_info().get_pids());
    if (info.get_cluster_id() == 0) {
        int agg_correct_count = 0;
        int agg_count = 0;
        while (mailbox->poll(0,0)) {
            auto bin = mailbox->recv(0,0);
            int p1, p2;
            bin >> p1 >> p2;
            agg_correct_count += p1;
            agg_count += p2;
        }
        husky::LOG_I << "Global test result: (" << agg_correct_count << "/" << agg_count << ")"
            << " Global ensemble accuracy: " << agg_correct_count*1.0/agg_count;
    }
}

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
    std::vector<int> kvs;
    for (int i = 0; i < 10; ++ i) {
        auto train_task = TaskFactory::Get().CreateTask<MLTask>();
        train_task.set_dimensions(config.num_params);
        train_task.set_total_epoch(config.train_epoch);  // set epoch number
        train_task.set_num_workers(config.num_train_workers);
        int kv1 = create_kvstore_and_set_hint(hint, train_task, config.num_params);  // Create KVStore and Set hint
        assert(kv1 != -1);
        kvs.push_back(kv1);
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
    if (Context::get_worker_info().get_process_id() == 0)
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
    if (Context::get_worker_info().get_process_id() == 0)
        husky::LOG_I << YELLOW("train time: " + std::to_string(train_time) + " ms");

    // Submit test_task
    auto test_task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>();
    test_task.set_worker_num({2});  // number of test threads per process 
    test_task.set_worker_num_type({"threads_per_worker"});
    engine.AddTask(test_task, [kvs, &data_store, &config](const Info& info){
        test_lambda(data_store, kvs, config.num_params, info);
    });
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto test_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    husky::LOG_I << YELLOW("test time: " + std::to_string(test_time) + " ms");

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
