#include <vector>

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
 */
int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", 
                                       "hdfs_namenode", "hdfs_namenode_port",
                                       "input", "num_features", "alpha", "num_iters",
                                       "train_epoch"});

    int train_epoch = std::stoi(Context::get_param("train_epoch"));
    float alpha = std::stof(Context::get_param("alpha"));
    int num_iters = std::stoi(Context::get_param("num_iters"));
    int num_features = std::stoi(Context::get_param("num_features"));
    int num_params = num_features + 1; // +1 for intercept
    
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());
    // Create the DataStore
    datastore::DataStore<LabeledPointHObj<float, float, true>> data_store(Context::get_worker_info().get_num_local_workers());

    auto task = TaskFactory::Get().CreateTask<HuskyTask>(1, 4); // 1 epoch, 1 workers
    engine.AddTask(std::move(task), [&data_store, &num_features](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, local_id);
    });


    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
    auto task1 = TaskFactory::Get().CreateTask<GenericMLTask>();
    task1.set_dimensions(num_params);
    task1.set_kvstore(kv1);
    task1.set_total_epoch(train_epoch);  // set epoch number
    task1.set_num_workers(1);   // set worker number
    // task1.set_running_type(Task::Type::HogwildTaskType);
    // task1.set_running_type(Task::Type::PSTaskType);
    task1.set_running_type(Task::Type::SingleTaskType);
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
        BatchDataSampler<LabeledPointHObj<float, float, true>> batch_data_sampler(data_store, 100);
        batch_data_sampler.random_start_point();
        for (int iter = 0; iter < num_iters; ++ iter) {
            // sgd_update(worker, data_sampler, alpha);
            batch_sgd_update(worker, batch_data_sampler, alpha, 100);
            // test model
            if (info.get_cluster_id() == 0) {
                DataIterator<LabeledPointHObj<float, float, true>> data_iterator(data_store);
                float test_error = get_test_error(worker, data_iterator, num_params);
                husky::LOG_I << "Iter:" << std::to_string(iter)<< " Accuracy is " << test_error;
            }
        }
    });
    engine.Submit();
    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
