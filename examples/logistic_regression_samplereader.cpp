#include <vector>
#include <chrono>

// #include "datastore/datastore.hpp"
#include "worker/engine.hpp"
#include "ml/common/mlworker.hpp"

#include "sample_reader.hpp"

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

template <template <typename> typename ContainerT>
void batch_sgd_update(const std::unique_ptr<ml::common::GenericMLWorker>& worker,
        ContainerT<LabeledPointHObj<float, float, true>>* container, float alpha) {
    alpha /= container->get_batch_size();
    auto keys = container->prepare_next_batch();
    worker->Prepare_v2(keys);
    for (auto data : container->get_data_ptrs()) {
        if (data == NULL) break;
        auto& x = data->x;
        float y = data->y;
        if (y < 0) y = 0;
        float pred_y = 0.0;

        int i = 0;
        for (auto iter = x.begin_feaval(); iter != x.end_feaval(); ++iter) {
            const auto& field = *iter;
            while (keys[i] < field.fea) i += 1;
            pred_y += worker->Get_v2(i) * field.val;
        }  // TODO add intercept
        pred_y = 1. / (1. + exp(-1 * pred_y)); 
        i = 0;
        for (auto iter = x.begin_feaval(); iter != x.end_feaval(); ++iter) {
            const auto& field = *iter;
            while (keys[i] < field.fea) i += 1;
            worker->Update_v2(i, alpha * field.val * (y - pred_y));
        }
    }
    worker->Clock_v2();
}

template <template <typename> typename ContainerT>
float get_test_error_v2(const std::unique_ptr<ml::common::GenericMLWorker>& worker, 
        ContainerT<LabeledPointHObj<float, float, true>>* container,
        int num_params, int test_samples = -1) {
    test_samples = 100;
    std::vector<husky::constants::Key> all_keys;
    for (int i = 0; i < num_params; i++) all_keys.push_back(i);
    worker->Prepare_v2(all_keys);
    int count = 0;
    float c_count = 0; //correct count
    bool has_next = true;
    while (count < test_samples && has_next) {
        container->prepare_next_batch();
        for (auto data : container->get_data_ptrs()) {
            if (data == NULL) {
                has_next = false;
                break;
            }
            count = count + 1;
            auto& x = data->x;
            float y = data->y;
            if (y < 0) y = 0;
            float pred_y = 0.0;
            for (auto iter = x.begin_feaval(); iter != x.end_feaval(); ++iter) {
                const auto& field = *iter;
                pred_y += worker->Get_v2(field.fea) * field.val;
            }
            pred_y = (pred_y > 0) ? 1 : 0;
            if (int(pred_y) == int(y)) { c_count += 1;}

            if (count == test_samples) break;
        }
    }
    worker->Clock_v2();
    return c_count/count;
}

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

    auto task1 = TaskFactory::Get().CreateTask<MLTask>();
    task1.set_dimensions(num_params);
    task1.set_total_epoch(train_epoch);  // set epoch number

    if (model == "Single") {
        assert(num_train_workers == 1);
        int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
        task1.set_num_workers(num_train_workers);
        task1.set_hint("single");
        task1.set_kvstore(kv1);
        husky::LOG_I << GREEN("Setting to Single, threads: "+std::to_string(num_train_workers));
    } else if (model == "Hogwild") {
        int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
        task1.set_num_workers(4);
        task1.set_hint("hogwild");
        husky::LOG_I << GREEN("Setting to Hogwild, threads: "+std::to_string(num_train_workers));
        task1.set_kvstore(kv1);
    } else if (model == "PSBSP") {
        int kv1 = kvstore::KVStore::Get().CreateKVStore<float>(kvstore::KVServerBSPHandle<float>(num_train_workers));
        task1.set_num_workers(num_train_workers);
        task1.set_hint("PS#BSP");
        task1.set_kvstore(kv1);
        husky::LOG_I << GREEN("Setting to PSBSP, threads: "+std::to_string(num_train_workers));
    } else if (model == "PSSSP") {
        int staleness = 2;
        int kv1 = kvstore::KVStore::Get().CreateKVStore<float>(kvstore::KVServerSSPHandle<float>(num_train_workers, staleness));
        task1.set_num_workers(num_train_workers);
        task1.set_hint("PS#SSP");
        task1.set_kvstore(kv1);
        husky::LOG_I << GREEN("Setting to PSSSP, threads: "+std::to_string(num_train_workers)+" Staleness: "+std::to_string(staleness));
    } else if (model == "PSASP") {
        int kv1 = kvstore::KVStore::Get().CreateKVStore<float>(kvstore::KVServerDefaultAddHandle<float>());  // use the default add handle
        task1.set_num_workers(num_train_workers);
        task1.set_hint("PS#ASP");
        task1.set_kvstore(kv1);
        husky::LOG_I << GREEN("Setting to PSASP, threads: "+std::to_string(num_train_workers));
    } else {
        husky::LOG_I << RED("Model error: "+model);
    }


    // create a TextBuffer
    int batch_size = 100, batch_num = 50;
    auto buffer = new TextBuffer(Context::get_param("input"), batch_size, batch_num);

    engine.AddTask(std::move(task1), [num_iters, alpha, num_params, num_features, buffer](const Info& info) {
        int batch_size = 100;
        // create a reader
        SampleReader<LabeledPointHObj<float,float,true>> * reader = new LIBSVMSampleReader<float, float, true>(batch_size, num_features, buffer);

        if (reader->is_empty()) {
            return;  // return if there's no data
        }
        auto& worker = info.get_mlworker();

        // main loop
        for (int iter = 0; iter < num_iters; ++ iter) {
            batch_sgd_update(worker, reader, alpha);
        }

        auto accuracy = get_test_error_v2(worker, reader, num_params);
        husky::LOG_I << "accuracy: " << accuracy;
        delete reader;
    });
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count();
    husky::LOG_I << YELLOW("Elapsed time: "+std::to_string(train_time) + " ms");

    engine.Exit();
    kvstore::KVStore::Get().Stop();
    delete buffer;
}
