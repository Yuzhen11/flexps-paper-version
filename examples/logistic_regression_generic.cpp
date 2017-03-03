#include <chrono>
#include <vector>

#include "datastore/datastore.hpp"
#include "datastore/datastore_utils.hpp"
#include "worker/engine.hpp"

#include "lib/load_data.hpp"
#include "lib/task_utils.hpp"

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
    bool rt =
        init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", "hdfs_namenode",
                                    "hdfs_namenode_port", "input", "num_features", "alpha", "num_iters", "train_epoch",
                                    "kType", "kConsistency", "num_train_workers", "num_load_workers", "trainer"});

    int train_epoch = std::stoi(Context::get_param("train_epoch"));
    float alpha = std::stof(Context::get_param("alpha"));
    int num_iters = std::stoi(Context::get_param("num_iters"));
    int num_features = std::stoi(Context::get_param("num_features"));
    int num_params = num_features + 1;  // +1 because starting from 1, but not for intercept
    std::string kType = Context::get_param("kType");
    std::string kConsistency = Context::get_param("kConsistency");
    int num_train_workers = std::stoi(Context::get_param("num_train_workers"));
    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    const std::string kTrainer = Context::get_param("trainer");
    const std::vector<std::string> trainers_set({"lr", "svm"});
    assert(std::find(trainers_set.begin(), trainers_set.end(), kTrainer) != trainers_set.end());
    husky::LOG_I << CLAY("trainer is set to "+kTrainer);

    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kType, kType},
        {husky::constants::kConsistency, kConsistency},
        {husky::constants::kNumWorkers, std::to_string(num_train_workers)},
        {husky::constants::kEnableDirectModelTransfer, "on"},
        {husky::constants::kStaleness, "1"}  // default stalenss
    };

    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());
    // Create the DataStore
    datastore::DataStore<LabeledPointHObj<float, float, true>> data_store(
        Context::get_worker_info().get_num_local_workers());

    auto task = TaskFactory::Get().CreateTask<HuskyTask>(1, num_load_workers);  // 1 epoch, 1 workers
    engine.AddTask(std::move(task), [&data_store, &num_features](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, local_id);
    });

    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    husky::LOG_I << YELLOW("Load time: " + std::to_string(load_time) + " ms");

    auto task1 = TaskFactory::Get().CreateTask<MLTask>();
    task1.set_dimensions(num_params);
    task1.set_total_epoch(train_epoch);  // set epoch number
    task1.set_num_workers(num_train_workers);
    // Create KVStore and Set hint
    int kv1 = create_kvstore_and_set_hint(hint, task1);
    assert(kv1 != -1);
    // Set max key, to make the keys distributed
    kvstore::RangeManager::Get().SetMaxKeyAndChunkSize(kv1, num_params);

    engine.AddTask(std::move(task1), [&data_store, kTrainer, num_iters, alpha, num_params](const Info& info) {
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
        for (int iter = 0; iter < num_iters; ++iter) {
            if (kTrainer == "lr") {
                // sgd_update(worker, data_sampler, alpha);
                lr::batch_sgd_update_lr(worker, batch_data_sampler, alpha);
            } else if (kTrainer == "svm") {
                svm::batch_sgd_update_svm_dense(worker, batch_data_sampler, alpha, num_params);
            }

            if (iter % 10 == 0) {
                // Testing, now all the threads need to run `get_test_error`, it is for PS.
                // So it won't mess up the iteration
                datastore::DataIterator<LabeledPointHObj<float, float, true>> data_iterator(data_store);
                float test_error = -1;
                if (kTrainer == "lr") {
                    test_error = lr::get_test_error_lr_v2(worker, data_iterator, num_params);
                } else if (kTrainer == "svm") {
                    test_error = svm::get_test_error_svm_v2(worker, data_iterator, num_params);
                }
                if (info.get_cluster_id() == 0) {
                    husky::LOG_I << "Iter:" << std::to_string(iter) << " Accuracy is " << test_error;
                }
            }
        }
    });
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    husky::LOG_I << YELLOW("train time: " + std::to_string(train_time) + " ms");

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
