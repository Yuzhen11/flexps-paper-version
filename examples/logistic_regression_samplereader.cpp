#include <vector>
#include <chrono>

#include "worker/engine.hpp"
#include "lib/sample_reader.hpp"
#include "lib/task_utils.hpp"

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
 * 
 * input=hdfs:///datasets/classification/a9
 * alpha=0.5
 * num_iters=10
 * num_features=123
 * train_epoch=1
 *
 */

template <template <typename> typename ContainerT>
void batch_sgd_update(const std::unique_ptr<ml::mlworker::GenericMLWorker>& worker,
        ContainerT<LabeledPointHObj<float, float, true>>* container, float alpha, int num_params) {
    alpha /= container->get_batch_size();
    auto keys = container->prepare_next_batch();
    keys.push_back(num_params - 1);
    worker->Prepare_v2(keys);
    for (auto data : container->get_data()) {
        auto& x = data.x;
        float y = data.y;
        if (y < 0) y = 0;
        float pred_y = 0.0;

        int i = 0;
        for (auto iter = x.begin_feaval(); iter != x.end_feaval(); ++iter) {
            const auto& field = *iter;
            while (i < keys.size() && keys[i] < field.fea) i += 1;
            assert(keys[i] == field.fea);
            pred_y += worker->Get_v2(i) * field.val;
        }
        while (i < keys.size() && keys[i] < num_params - 1) i += 1;
        assert(keys[i] == num_params - 1);
        pred_y += worker->Get_v2(i);  // intercept

        pred_y = 1. / (1. + exp(-1 * pred_y)); 

        worker->Update_v2(i, alpha * (y - pred_y));  // intercept
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
float get_test_error_v2(const std::unique_ptr<ml::mlworker::GenericMLWorker>& worker, 
        ContainerT<LabeledPointHObj<float, float, true>>* container,
        int num_params, int test_samples = -1) {
    test_samples = 100;
    std::vector<husky::constants::Key> all_keys;
    for (int i = 0; i < num_params; i++) all_keys.push_back(i);
    worker->Prepare_v2(all_keys);
    int count = 0;
    float c_count = 0; //correct count
    while (count < test_samples) {
        container->prepare_next_batch();
        auto batch = container->get_data();
        if (batch.size() == 0) break;
        for (auto data : batch) {
            count = count + 1;
            auto& x = data.x;
            float y = data.y;
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
    assert(count != 0);
    return c_count/count;
}

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", 
                                       "hdfs_namenode", "hdfs_namenode_port",
                                       "input", "num_features", "alpha", "num_iters",
                                       "train_epoch",
                                       "kType", "kConsistency", "num_train_workers"});

    int train_epoch = std::stoi(Context::get_param("train_epoch"));
    float alpha = std::stof(Context::get_param("alpha"));
    int num_iters = std::stoi(Context::get_param("num_iters"));
    int num_features = std::stoi(Context::get_param("num_features"));
    int num_params = num_features + 1; // +1 for intercept
    std::string kType = Context::get_param("kType");
    std::string kConsistency = Context::get_param("kConsistency");
    int num_train_workers = std::stoi(Context::get_param("num_train_workers"));
    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kType, kType},
        {husky::constants::kConsistency, kConsistency},
        {husky::constants::kNumWorkers, std::to_string(num_train_workers)}
    };
    
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    auto task1 = TaskFactory::Get().CreateTask<MLTask>();
    task1.set_dimensions(num_params);
    task1.set_total_epoch(train_epoch);  // set epoch number
    task1.set_num_workers(num_train_workers);
    // Create KVStore and Set hint
    int kv1 = create_kvstore_and_set_hint(hint, task1, num_params);
    assert(kv1 != -1);

    // create a AsyncReadBuffer
    int batch_size = 100, batch_num = 50;
    AsyncReadBuffer buffer;
    engine.AddTask(std::move(task1), [num_train_workers, batch_size, batch_num, num_iters, alpha, num_params, num_features, &buffer](const Info& info) {
        buffer.init(Context::get_param("input"), info.get_task_id(), num_train_workers, batch_size, batch_num);
        int batch_size = 100;
        // create a reader
        std::unique_ptr<SampleReader<LabeledPointHObj<float,float,true>>> reader(new LIBSVMSampleReader<float, float, true>(batch_size, num_features, &buffer));

        if (reader->is_empty()) return;  // return if there's no data
        auto& worker = info.get_mlworker();

        // main loop
        for (int iter = 0; iter < num_iters; ++ iter) {
            batch_sgd_update(worker, reader.get(), alpha, num_params);
        }

        auto accuracy = get_test_error_v2(worker, reader.get(), num_params);
        husky::LOG_I << "accuracy: " << accuracy;
    });
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count();
    husky::LOG_I << YELLOW("Elapsed time: "+std::to_string(train_time) + " ms");

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
