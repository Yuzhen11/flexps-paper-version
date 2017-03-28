#include <chrono>
#include <vector>

#include "worker/engine.hpp"
#include "ml/ml.hpp"
#include "lib/sample_reader_parse.hpp"
#include "lib/task_utils.hpp"
#include "lib/app_config.hpp"
#include "io/input/line_inputformat_ml.hpp"
#include "io/input/binary_inputformat_ml.hpp"

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
 * num_iters=10  # -1 means use all data
 * num_features=123
 * train_epoch=1
 * batch_num=3
 * batch_size=20
 *
 */

float log_loss(float label, float predict) {
    if (label < 0.5) {  // label is 0
        predict = 1.0f - predict;
    }
    return -log(predict < 0.000001f ? 0.000001f : predict);
}

template <typename ContainerT>
int batch_sgd_update(const std::unique_ptr<ml::mlworker::GenericMLWorker<float>>& worker,
        ContainerT* container, float alpha, int num_params, float& loss) {
    alpha /= container->get_batch_size();
    auto keys = container->prepare_next_batch();
    keys.push_back(num_params - 1);
    worker->Prepare_v2(keys);
    auto& data_batch = container->get_data();
    if (data_batch.empty()) return 0;
    for (auto data : data_batch) {
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
        loss += log_loss(y, pred_y);

        worker->Update_v2(i, alpha * (y - pred_y));  // intercept
        i = 0;
        for (auto iter = x.begin_feaval(); iter != x.end_feaval(); ++iter) {
            const auto& field = *iter;
            while (keys[i] < field.fea) i += 1;
            worker->Update_v2(i, alpha * field.val * (y - pred_y));
        }
    }
    worker->Clock_v2();

    return data_batch.size();
}

template <typename ContainerT>
float get_test_error_v2(const std::unique_ptr<ml::mlworker::GenericMLWorker<float>>& worker, 
        ContainerT* container,
        int num_params, int test_samples = -1) {
    std::vector<husky::constants::Key> all_keys;
    for (int i = 0; i < num_params; i++) all_keys.push_back(i);
    worker->Prepare_v2(all_keys);
    int count = 0;
    float c_count = 0; //correct count
    while (count < test_samples || test_samples == -1) {
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

    auto task1 = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>();
    if (config.num_train_workers == 1 && config.kType == husky::constants::kSingle && config.kLoadHdfsType == "load_hdfs_locally") {
        task1.set_worker_num({1});
        task1.set_worker_num_type({"threads_traverse_cluster"});
    } else {
        task1.set_type(husky::Task::Type::MLTaskType);
    }
    task1.set_dimensions(config.num_params);
    task1.set_total_epoch(config.train_epoch);  // set epoch number
    task1.set_num_workers(config.num_train_workers);
    // Create KVStore and Set hint
    int kv1 = create_kvstore_and_set_hint(hint, task1, config.num_params);
    assert(kv1 != -1);

    // create a AsyncReadBuffer
    int batch_size = 20, batch_num = 3;

    int is_binary = config.is_binary;

    // binary
    LIBSVMAsyncReadBinaryParseBuffer<LabeledPointHObj<float, float, true>, io::BinaryInputFormatML> buffer_binary;
    LIBSVMAsyncReadParseBuffer<LabeledPointHObj<float, float, true>, io::LineInputFormatML> buffer;
    if (is_binary) {
        engine.AddTask(std::move(task1), [config, batch_size, batch_num, &buffer_binary](const Info& info) {
            buffer_binary.init(Context::get_param("input"), info.get_task_id(), config.num_train_workers, batch_size, batch_num, config.num_features);
            // create a reader
            
            std::unique_ptr<SimpleSampleReader<LabeledPointHObj<float,float,true>, io::BinaryInputFormatML>> reader(new SimpleSampleReader<LabeledPointHObj<float, float, true>, io::BinaryInputFormatML>(&buffer_binary));

            if (reader->is_empty()) {
                husky::LOG_I << "no data";  // for debug
                return;  // return if there's no data
            }

            auto worker = ml::CreateMLWorker<float>(info);

            float train_loss = 0.0f;
            int sample_count = 0;
            int report_interval = 10000;
            int sample_total = 0;
            // main loop
            for (int iter = 0; iter < config.num_iters || config.num_iters == -1; ++ iter) {
                sample_count += batch_sgd_update(worker, reader.get(), config.alpha, config.num_params, train_loss);
                if (reader->is_empty()) break;
                if (sample_count >= report_interval) {
                    sample_total += sample_count;
                    husky::LOG_I << "train loss " << (train_loss / sample_count);
                    husky::LOG_I << "samples seen " << sample_total;
                    train_loss = 0.0f;
                    sample_count = 0;
                }
            }

            sample_total += sample_count;
            husky::LOG_I << "total training samples in phase<binary> " << info.get_current_epoch() << ": " << sample_total;

            auto accuracy = get_test_error_v2(worker, reader.get(), config.num_params);
            husky::LOG_I << "accuracy: " << accuracy;
        });
    } else {
        engine.AddTask(std::move(task1), [config, batch_size, batch_num, &buffer](const Info& info) {
            buffer.init(Context::get_param("input"), info.get_task_id(), config.num_train_workers, batch_size, batch_num, config.num_features);
            // create a reader
            std::unique_ptr<SimpleSampleReader<LabeledPointHObj<float,float,true>, io::LineInputFormatML>> reader(new SimpleSampleReader<LabeledPointHObj<float, float, true>, io::LineInputFormatML>(&buffer));

            if (reader->is_empty()) {
                husky::LOG_I << "no data";  // for debug
                return;  // return if there's no data
            }

            auto worker = ml::CreateMLWorker<float>(info);

            float train_loss = 0.0f;
            int sample_count = 0;
            int report_interval = 10000;
            int sample_total = 0;
            // main loop
            for (int iter = 0; iter < config.num_iters || config.num_iters == -1; ++ iter) {
                sample_count += batch_sgd_update(worker, reader.get(), config.alpha, config.num_params, train_loss);
                if (reader->is_empty()) break;
                if (sample_count >= report_interval) {
                    sample_total += sample_count;
                    husky::LOG_I << "train loss " << (train_loss / sample_count);
                    husky::LOG_I << "samples seen " << sample_total;
                    train_loss = 0.0f;
                    sample_count = 0;
                }
            }

            sample_total += sample_count;
            husky::LOG_I << "total training samples in phase<not_binary> " << info.get_current_epoch() << ": " << sample_total;

            // auto accuracy = get_test_error_v2(worker, reader.get(), config.num_params);
            // husky::LOG_I << "accuracy: " << accuracy;
        });
    }

    // Submit Task
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count();
    husky::LOG_I << YELLOW("Elapsed time: "+std::to_string(train_time) + " ms");

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
