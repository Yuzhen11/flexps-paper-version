#include <chrono>
#include <vector>

#include "worker/engine.hpp"
#include "ml/ml.hpp"
#include "lib/sample_reader_parse.hpp"
#include "lib/task_utils.hpp"
#include "lib/app_config.hpp"
#include "io/input/line_inputformat_ml.hpp"
#include "io/input/binary_inputformat_ml.hpp"

/*
 * This is to show that in a machine learning workload, the data loading will take a large portion
 * of time. Don't use any AsyncBuffer.
 * 
 * only support binary
 */

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

float log_loss(float label, float predict) {
    if (label < 0.5) {  // label is 0
        predict = 1.0f - predict;
    }
    return -log(predict < 0.000001f ? 0.000001f : predict);
}

int batch_sgd_update(const std::unique_ptr<ml::mlworker::GenericMLWorker<float>>& worker,
        const std::vector<LabeledPointHObj<float, float, true>>& batch, 
        std::vector<husky::constants::Key>& keys, 
        float alpha, int num_params, float& loss) {
    alpha /= batch.size();
    keys.push_back(num_params - 1);
    worker->Prepare_v2(keys);
    for (auto data : batch) {
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
    return batch.size();
}

int main(int argc, char** argv) {
    // Set config
    config::InitContext(argc, argv, {"is_binary", "kLoadHdfsType"});
    auto config = config::SetAppConfigWithContext();
    if (Context::get_worker_info().get_process_id() == 0)
        config:: ShowConfig(config);
    auto hint = config::ExtractHint(config);

    auto& engine = Engine::Get();
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    auto task1 = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>();
    if (config.num_train_workers == 1 && config.kType == husky::constants::kSingle && Context::get_param("kLoadHdfsType") == "load_hdfs_locally") {
        task1.set_worker_num({1});
        task1.set_worker_num_type({"threads_traverse_cluster"});
    } else {
        task1.set_type(husky::Task::Type::BasicTaskType);
    }
    task1.set_dimensions(config.num_params);
    task1.set_total_epoch(config.train_epoch);  // set epoch number
    task1.set_num_workers(config.num_train_workers);
    // Create KVStore and Set hint
    int kv1 = create_kvstore_and_set_hint(hint, task1, config.num_params);
    assert(kv1 != -1);

    bool is_binary = Context::get_param("is_binary") == "on" ? true:false;
    assert(is_binary == true);  // only support binary format for simplicity
    engine.AddTask(task1, [task1, config](const Info& info){
        auto mlworker = ml::CreateMLWorker<float>(info);
        husky::io::BinaryInputFormatML infmt(Context::get_param("input"), config.num_train_workers, task1.get_id());
        typename husky::io::BinaryInputFormatML::RecordT record;  // BinaryInputFormatRecord
        std::vector<LabeledPointHObj<float, float, true>> batch;
        std::vector<husky::constants::Key> indexes;
        std::vector<husky::constants::Key> tmp;
        std::vector<husky::constants::Key> tmp2;
        int batch_size = 100;
        float train_loss = 0;
        int sample_count = 0;
        int report_interval = 10000;
        int sample_total = 0;
        while (infmt.next(record)) {
            husky::base::BinStream& bin = husky::io::BinaryInputFormatML::recast(record);
            float y;
            std::vector<std::pair<int, float>> v;
            while (bin.size()) {
                bin >> y >> v;
                // Generate one data
                LabeledPointHObj<float, float, true> data(config.num_features);
                data.y = y;
                // tmp.clear();
                // tmp.reserve(v.size());
                for (auto p : v) {
                    data.x.set(p.first-1, p.second);
                    // tmp.push_back(p.first-1);
                    indexes.push_back(p.first-1);
                }
                // for std::merge
                // tmp2.resize(tmp.size() + indexes.size());
                // std::merge(tmp.begin(), tmp.end(), indexes.begin(), indexes.end(), tmp2.begin());
                // tmp2.swap(indexes);
                // indexes.erase(std::unique(indexes.begin(), indexes.end()), indexes.end());
                batch.push_back(std::move(data));  // push into batch
                // If batch is full, handle it
                if (batch.size() == batch_size) {
                    std::sort(indexes.begin(), indexes.end());
                    indexes.erase(std::unique(indexes.begin(), indexes.end()), indexes.end());
                    // husky::LOG_I << "batch_sgd_update...";
                    sample_count += batch_sgd_update(mlworker, batch, indexes, config.alpha, config.num_params, train_loss);
                    // husky::LOG_I << "train_loss: " << train_loss / sample_count;
                    if (sample_count >= report_interval) {
                        sample_total += sample_count;
                        husky::LOG_I << "train loss " << (train_loss / sample_count);
                        husky::LOG_I << "samples seen " << sample_total;
                        train_loss = 0.0f;
                        sample_count = 0;
                    }
                    // clear
                    indexes.clear();
                    batch.clear();
                }
            }
        }
        husky::LOG_I << "sample_total: " << sample_total;
    });

    // Submit Task
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count();
    husky::LOG_I << YELLOW("Elapsed time: "+std::to_string(train_time) + " ms");

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
