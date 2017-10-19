#pragma once

#include <fstream>
#include <cmath>

#include "core/info.hpp"
#include "datastore/datastore.hpp"
#include "datastore/datastore_utils.hpp"
#include "lib/app_config.hpp"
#include "lib/lr_updater.hpp"
#include "ml/ml.hpp"

#include "husky/io/input/line_inputformat.hpp"
#include "husky/lib/vector.hpp"
#include "husky/lib/ml/feature_label.hpp"

namespace husky {
namespace lr {
namespace {

// Perform lr_test
bool lr_test_accuracy(const husky::lib::ml::LabeledPointHObj<float, float, true>& data, const std::vector<float>& model) {
    auto& x = data.x;
    float y = data.y;
    if (y < 0)
        y = 0;
    float pred_y = 0.0;
    for (auto field : x) {
        pred_y += model[field.fea] * field.val;
    }
    pred_y += model[model.size() - 1];  // intercept
    pred_y = 1. / (1. + exp(-pred_y));
    pred_y = (pred_y > 0.5) ? 1 : 0;
    return int(pred_y) == int(y) ? true : false;
}

float lr_test_objective(const husky::lib::ml::LabeledPointHObj<float, float, true>& data, const std::vector<float>& model) {
    auto& x = data.x;
    float y = data.y;
    if (y < 0) y = 0;
    float pred_y = 0.0f;
    for (auto& field : x) {
        pred_y += model[field.fea] * field.val;
    }
    pred_y += model[model.size() - 1];  // intercept
    pred_y = 1. / (1. + exp(-pred_y));
    if (y == 0) {
        return -log(1. - pred_y);
    } else {  // y == 1
        return -log(pred_y);
    }
}

float squared_deviation(const std::vector<float>& model1, const std::vector<float>& model2) {
    int num_params = model1.size();
    assert(num_params == model2.size());
    float sqr_dev = 0.0f;
    for (int i = 0; i < num_params; ++i) {
        auto diff = std::fabs(model1[i] - model2[i]);
        sqr_dev += diff * diff;
    }
    sqr_dev /= num_params;
    return sqr_dev;
}

void load_benchmark_model(std::vector<float>& benchmark_model, const std::string& model_file, int num_params) {
    benchmark_model.resize(num_params);
    // auto& infmt = husky::io::InputFormatStore::create_line_inputformat();
    husky::io::LineInputFormat infmt;
    infmt.set_input(model_file);
    typename io::LineInputFormat::RecordT record;
    bool success = false;
    int idx = 0;
    while (true) {
        success = infmt.next(record);
        if (success == false) break;
        if(record.find(",") != boost::string_ref::npos) {
            char* record_pos = NULL;
            int length = record.size();
            std::unique_ptr<char> record_ptr(new char[length + 1]);
            strncpy(record_ptr.get(), record.data(), length);
            record_ptr.get()[length] = '\0';
            char* tok = strtok_r(record_ptr.get(), " ,", &record_pos);

            while (tok != NULL) {
                assert(idx < benchmark_model.size());
                benchmark_model[idx] = std::atof(tok);
                idx += 1;
                tok = strtok_r(NULL, " ,", &record_pos);
            }
        } else {
            assert(idx < num_params);
            benchmark_model[idx] = std::stof(record.to_string());
            idx += 1;
        }
    }
}

void sgd_train(const Info& info, datastore::DataStore<LabeledPointHObj<float,float,true>>& data_store, const config::AppConfig& config, std::vector<float>& benchmark_model, int report_interval, int accum_iter = 0, bool write_model = false) {
    auto worker = ml::CreateMLWorker<float>(info);
    // Create BatchDataSampler for mini-batch SGD
    datastore::BatchDataSampler<husky::lib::ml::LabeledPointHObj<float, float, true>> batch_data_sampler(data_store, config.batch_size);
    batch_data_sampler.random_start_point();

    auto start_train = std::chrono::steady_clock::now();
    long long test_time = 0;
    for (int iter = accum_iter; iter < config.num_iters + accum_iter; ++iter) {
        // Report deviations
        if (iter % report_interval == 0) {
            std::vector<float> vals;
            if (info.get_cluster_id() == 0) {
                auto current_time = std::chrono::steady_clock::now();
                auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_train).count();
                train_time -= test_time;

                // Pull trained model
                std::vector<husky::constants::Key> keys(config.num_params);
                for (int i = 0; i < config.num_params; ++i) keys[i] = i;
                worker->Pull(keys, &vals);
                worker->Push({keys[0]}, {0});

                if (!benchmark_model.empty()) {
                    // Test with true model
                    float sqr_dev = lr::squared_deviation(benchmark_model, vals);

                    auto test_end = std::chrono::steady_clock::now();
                    test_time += std::chrono::duration_cast<std::chrono::milliseconds>(test_end - current_time).count();
                    husky::LOG_I << "Task " << info.get_task_id() << ": Iter, Time, SqrDev: " << iter << "," << train_time << "," << std::setprecision(15) << sqr_dev;
                } else {
                    // Test with training samples
                    datastore::DataIterator<LabeledPointHObj<float, float, true>> data_iterator(data_store);
                    int count = 0;
                    float loss = 0.0f;
                    while (data_iterator.has_next()) {
                        auto& data = data_iterator.next();
                        loss += lr::lr_test_objective(data, vals);
                        count += 1;
                    }
                    loss /= count;

                    auto test_end = std::chrono::steady_clock::now();
                    test_time += std::chrono::duration_cast<std::chrono::milliseconds>(test_end - current_time).count();
                    husky::LOG_I << "Task " << info.get_task_id() << ": Iter, Time, Loss: " << iter << "," << train_time << "," << std::setprecision(15) << loss;
                }
            } else {
                worker->Pull({0}, &vals);
                worker->Push({0}, {0});
            }
        }

        float alpha = config.alpha / (iter / (int)config.learning_rate_coefficient + 1);
        alpha = std::max(1e-5f, alpha);
        lr::batch_sgd_update_lr(worker, batch_data_sampler, alpha);
    }
    // last report of this stage
    std::vector<float> vals;
    if (info.get_cluster_id() == 0) {
        auto current_time = std::chrono::steady_clock::now();
        auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_train).count();
        train_time -= test_time;

        // Pull trained model
        std::vector<husky::constants::Key> keys(config.num_params);
        for (int i = 0; i < config.num_params; ++i) keys[i] = i;
        worker->Pull(keys, &vals);
        worker->Push({keys[0]}, {0});

        if (!benchmark_model.empty()) {
            // Test with true model
            float sqr_dev = lr::squared_deviation(benchmark_model, vals);

            husky::LOG_I << "Task " << info.get_task_id() << ": Iter, Time, SqrDev: " << (config.num_iters + accum_iter - 1) << "," << train_time << "," << std::setprecision(15) << sqr_dev;
        } else {
            // Test with training samples
            datastore::DataIterator<LabeledPointHObj<float, float, true>> data_iterator(data_store);
            int count = 0;
            float loss = 0.0f;
            while (data_iterator.has_next()) {
                auto& data = data_iterator.next();
                loss += lr::lr_test_objective(data, vals);
                count += 1;
            }
            loss /= count;

            husky::LOG_I << "Task " << info.get_task_id() << ": Iter, Time, Loss: " << (config.num_iters + accum_iter - 1) << "," << train_time << "," << std::setprecision(15) << loss;
        }
    } else {
        worker->Pull({0}, &vals);
        worker->Push({0}, {0});
    }
    // output the model
    if (write_model && info.get_current_epoch() == info.get_total_epoch() - 1) {
        std::vector<husky::constants::Key> all_keys;
        for (int i = 0; i < config.num_params; i++)
            all_keys.push_back(i);
        worker->Prepare_v2(all_keys);
        if (info.get_cluster_id() == 0) {
            std::ofstream out;
            out.open("/data/tati/husky-45123/kdd.model_v" + std::to_string(info.get_task_id()));
            husky::LOG_I << "writing";
            for (int i = 0; i < config.num_params; ++ i) {
                out << worker->Get_v2(i) << "\n";
            }
            husky::LOG_I << "write done";
            out.close();
        }
        worker->Clock_v2();
    }
}

}  // namespace anonymous
}  // namespace lr
}  // namespace husky
