#include <vector>
#include <chrono>

#include "datastore/datastore.hpp"
#include "worker/engine.hpp"
#include "ml/common/mlworker.hpp"

#include "lib/load_data.hpp"
#include "lib/data_sampler.hpp"

#include "examples/updater.hpp"

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
 * ### Model should be Single/Hogwild/PSBSP, PSSSP, PSASP
 * model=PSBSP
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
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", 
                                       "hdfs_namenode", "hdfs_namenode_port",
                                       "input", "num_features", "alpha", "num_iters",
                                       "train_epoch",
                                       "model", "num_train_workers", "num_load_workers"});

    int train_epoch = std::stoi(Context::get_param("train_epoch"));
    float alpha = std::stof(Context::get_param("alpha"));
    int num_iters = std::stoi(Context::get_param("num_iters"));
    int num_features = std::stoi(Context::get_param("num_features"));
    int num_params = num_features + 1; // +1 for intercept
    std::string model = Context::get_param("model");
    int num_train_workers = std::stoi(Context::get_param("num_train_workers"));
    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());
    // Create the DataStore
    datastore::DataStore<LabeledPointHObj<float, float, true>> data_store(Context::get_worker_info().get_num_local_workers());

    auto task = TaskFactory::Get().CreateTask<HuskyTask>(1, num_load_workers); // 1 epoch, 1 workers
    engine.AddTask(std::move(task), [&data_store, &num_features](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, local_id);
    });

    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count();
    husky::LOG_I << YELLOW("Load time: "+std::to_string(load_time) + " ms");


    auto task1 = TaskFactory::Get().CreateTask<GenericMLTask>();
    task1.set_dimensions(num_params);
    task1.set_total_epoch(train_epoch);  // set epoch number

    if (model == "Single") {
        assert(num_train_workers == 1);
        int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
        task1.set_num_workers(num_train_workers);
        task1.set_running_type(Task::Type::SingleTaskType);
        task1.set_kvstore(kv1);
        husky::LOG_I << GREEN("Setting to Single, threads: "+std::to_string(num_train_workers));
    } else if (model == "Hogwild") {
        int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
        task1.set_num_workers(4);
        task1.set_running_type(Task::Type::HogwildTaskType);
        husky::LOG_I << GREEN("Setting to Hogwild, threads: "+std::to_string(num_train_workers));
        task1.set_kvstore(kv1);
    } else if (model == "PSBSP") {
        int kv1 = kvstore::KVStore::Get().CreateKVStore<float>(kvstore::KVServerBSPHandle<float>(num_train_workers));
        task1.set_num_workers(num_train_workers);
        task1.set_running_type(Task::Type::PSBSPTaskType);
        task1.set_kvstore(kv1);
        husky::LOG_I << GREEN("Setting to PSBSP, threads: "+std::to_string(num_train_workers));
    } else if (model == "PSSSP") {
        int staleness = 2;
        int kv1 = kvstore::KVStore::Get().CreateKVStore<float>(kvstore::KVServerSSPHandle<float>(num_train_workers, staleness));
        task1.set_staleness(staleness);
        // task1.set_worker_type("SSP");
        task1.set_num_workers(num_train_workers);
        task1.set_running_type(Task::Type::PSSSPTaskType);
        task1.set_kvstore(kv1);
        husky::LOG_I << GREEN("Setting to PSSSP, threads: "+std::to_string(num_train_workers)+" Staleness: "+std::to_string(staleness));
    } else if (model == "PSASP") {
        int kv1 = kvstore::KVStore::Get().CreateKVStore<float>(kvstore::KVServerDefaultAddHandle<float>());  // use the default add handle
        task1.set_num_workers(num_train_workers);
        task1.set_running_type(Task::Type::PSASPTaskType);
        task1.set_kvstore(kv1);
        husky::LOG_I << GREEN("Setting to PSASP, threads: "+std::to_string(num_train_workers));
    } else {
        husky::LOG_I << RED("Model error: "+model);
    }


    engine.AddTask(std::move(task1), [&data_store, num_iters, alpha, num_params](const Info& info) {
        // create a DataStoreWrapper
        DataStoreWrapper<LabeledPointHObj<float, float, true>> data_store_wrapper(data_store);
        if (data_store_wrapper.get_data_size() == 0) {
            return;  // return if there's not data
        }
        auto& worker = info.get_mlworker();
        // Create a DataSampler for SGD
        DataSampler<LabeledPointHObj<float, float, true>> data_sampler(data_store);
        data_sampler.random_start_point();
        // Create BatchDataSampler for mini-batch SGD
        int batch_size = 100;
        BatchDataSampler<LabeledPointHObj<float, float, true>> batch_data_sampler(data_store, batch_size);
        batch_data_sampler.random_start_point();
        for (int iter = 0; iter < num_iters; ++ iter) {
            // sgd_update_v2(worker, data_sampler, alpha);
            batch_sgd_update(worker, batch_data_sampler, alpha);

            if (iter % 10 == 0) {
                // Testing, now all the threads need to run `get_test_error`, it is for PS.
                // So it won't mess up the iteration
                DataIterator<LabeledPointHObj<float, float, true>> data_iterator(data_store);
                float test_error = get_test_error_v2(worker, data_iterator, num_params);
                if (info.get_cluster_id() == 0) {
                    husky::LOG_I << "Iter:" << std::to_string(iter)<< " Accuracy is " << test_error;
                }
            }
        }
    });
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count();
    husky::LOG_I << YELLOW("Load time: "+std::to_string(train_time) + " ms");

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
