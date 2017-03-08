#include <chrono>
#include <vector>

#include "datastore/datastore.hpp"
#include "datastore/datastore_utils.hpp"
#include "worker/engine.hpp"

#include "lib/load_data.hpp"
#include "lib/task_utils.hpp"
#include "lib/app_config.hpp"

#include "examples/lr_updater.hpp"
#include "examples/svm_updater.hpp"

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
    auto train_task = TaskFactory::Get().CreateTask<MLTask>();
    train_task.set_dimensions(config.num_params);
    train_task.set_total_epoch(config.train_epoch);  // set epoch number
    train_task.set_num_workers(config.num_train_workers);
    // Create KVStore and Set hint
    int kv1 = create_kvstore_and_set_hint(hint, train_task, config.num_params);
    assert(kv1 != -1);
    auto train_task_lambda = [&data_store, config](const Info& info) {
        // create a DataStoreWrapper
        datastore::DataStoreWrapper<LabeledPointHObj<float, float, true>> data_store_wrapper(data_store);
        if (data_store_wrapper.get_data_size() == 0) {
            return;  // return if there's not data
        }
        auto& worker = info.get_mlworker();
        // Create a DataSampler for SGD
        datastore::DataSampler<LabeledPointHObj<float, float, true>> data_sampler(data_store);
        data_sampler.random_start_point();
        // Create BatchDataSampler for mini-batch SGD
        int batch_size = 100;
        datastore::BatchDataSampler<LabeledPointHObj<float, float, true>> batch_data_sampler(data_store, batch_size);
        batch_data_sampler.random_start_point();
        for (int iter = 0; iter < config.num_iters; ++iter) {
            if (config.trainer == "lr") {
                // sgd_update(worker, data_sampler, config.alpha);
                lr::batch_sgd_update_lr(worker, batch_data_sampler, config.alpha);
            } else if (config.trainer == "svm") {
                svm::batch_sgd_update_svm_dense(worker, batch_data_sampler, config.alpha, config.num_params);
            }

            if (iter % 10 == 0) {
                // Testing, now all the threads need to run `get_test_error`, it is for PS.
                // So it won't mess up the iteration
                datastore::DataIterator<LabeledPointHObj<float, float, true>> data_iterator(data_store);
                float test_error = -1;
                if (config.trainer == "lr") {
                    test_error = lr::get_test_error_lr_v2(worker, data_iterator, config.num_params);
                } else if (config.trainer == "svm") {
                    test_error = svm::get_test_error_svm_v2(worker, data_iterator, config.num_params);
                }
                if (info.get_cluster_id() == 0) {
                    husky::LOG_I << "Iter:" << std::to_string(iter) << " Accuracy is " << test_error;
                }
            }
        }
    };

    // Submit load_task;
    engine.AddTask(std::move(load_task), load_task_lambda); 
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    husky::LOG_I << YELLOW("Load time: " + std::to_string(load_time) + " ms");

    // Submit train_task
    engine.AddTask(std::move(train_task), train_task_lambda);
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    husky::LOG_I << YELLOW("train time: " + std::to_string(train_time) + " ms");

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
