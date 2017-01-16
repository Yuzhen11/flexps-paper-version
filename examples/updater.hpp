#pragma once

#include "lib/data_sampler.hpp"
#include "ml/common/mlworker.hpp"

using husky::lib::ml::LabeledPointHObj;

namespace husky {
namespace {

// The SGD updater
void sgd_update(const std::unique_ptr<ml::common::GenericMLWorker>& worker, 
        DataSampler<LabeledPointHObj<float, float, true>>& data_sampler, 
        float alpha) {
    auto& data = data_sampler.next();  // get next data
    auto& x = data.x;
    float y = data.y;
    if (y < 0) y = 0;
    std::vector<int> keys;
    std::vector<float> params;
    std::vector<float> delta;
    keys.reserve(x.get_feature_num()+1);
    delta.reserve(x.get_feature_num()+1);
    for (auto field : x) {  // set keys
        keys.push_back(field.fea);
    }
    worker->Pull(keys, &params);  // issue Pull
    float pred_y = 0.0;
    int i = 0;
    for (auto field : x) {
        pred_y += params[i++] * field.val;
    }
    pred_y = 1. / (1. + exp(-1 * pred_y)); 
    i = 0;
    for (auto field : x) {
        delta.push_back(alpha * field.val * (y - pred_y));
        i += 1;
    }
    worker->Push(keys, delta);  // issue Push
};

// The SGD updater for v2 APIs
void sgd_update_v2(const std::unique_ptr<ml::common::GenericMLWorker>& worker, 
        DataSampler<LabeledPointHObj<float, float, true>>& data_sampler, 
        float alpha) {
    auto& data = data_sampler.next();  // get next data
    auto& x = data.x;
    float y = data.y;
    if (y < 0) y = 0;
    std::vector<int> keys;
    keys.reserve(x.get_feature_num()+1);
    for (auto field : x) {  // set keys
        keys.push_back(field.fea);
    }
    worker->Prepare_v2(keys);
    float pred_y = 0.0;
    int i = 0;
    for (auto field : x) {
        pred_y += worker->Get_v2(i++) * field.val;
    }
    pred_y = 1. / (1. + exp(-1 * pred_y)); 
    i = 0;
    for (auto field : x) {
        worker->Update_v2(i, alpha * field.val * (y - pred_y));
        i += 1;
    }
};

// The mini-batch SGD updator
void batch_sgd_update(const std::unique_ptr<ml::common::GenericMLWorker>& worker,
        BatchDataSampler<LabeledPointHObj<float, float, true>>& batch_data_sampler, float alpha, 
        int batch_size) {
    alpha /= batch_data_sampler.get_batch_size();
    std::vector<int> keys = batch_data_sampler.prepare_next_batch();  // prepare all the indexes in the batch
    std::vector<float> params;
    std::vector<float> delta;
    delta.resize(keys.size(), 0.0);
    worker->Pull(keys, &params);  // issue Pull
    for (auto data : batch_data_sampler.get_data_ptrs()) {  // iterate over the data in the batch
        auto& x = data->x;
        float y = data->y;
        if (y < 0) y = 0;
        float pred_y = 0.0;
        int i = 0;
        for (auto field : x) {
            while (keys[i] < field.fea) i += 1;
            pred_y += params[i] * field.val;
        }
        pred_y = 1. / (1. + exp(-1 * pred_y)); 
        i = 0;
        for (auto field : x) {
            while (keys[i] < field.fea) i += 1;
            delta[i] += alpha * field.val * (y - pred_y);
        }
    }
    worker->Push(keys, delta);  // issue Push
};

// The mini-batch SGD updator for v2
void batch_sgd_update_v2(const std::unique_ptr<ml::common::GenericMLWorker>& worker,
        BatchDataSampler<LabeledPointHObj<float, float, true>>& batch_data_sampler, float alpha, 
        int batch_size) {
    alpha /= batch_data_sampler.get_batch_size();
    std::vector<int> keys = batch_data_sampler.prepare_next_batch();  // prepare all the indexes in the batch
    worker->Prepare_v2(keys);
    for (auto data : batch_data_sampler.get_data_ptrs()) {  // iterate over the data in the batch
        auto& x = data->x;
        float y = data->y;
        if (y < 0) y = 0;
        float pred_y = 0.0;
        int i = 0;
        for (auto field : x) {
            while (keys[i] < field.fea) i += 1;
            pred_y += worker->Get_v2(i) * field.val;
        }
        pred_y = 1. / (1. + exp(-1 * pred_y)); 
        i = 0;
        for (auto field : x) {
            while (keys[i] < field.fea) i += 1;
            worker->Update_v2(i, alpha * field.val * (y - pred_y));
        }
    }
};

float get_test_error(const std::unique_ptr<ml::common::GenericMLWorker>& worker, 
        DataIterator<LabeledPointHObj<float, float, true>> data_iterator,
        int num_params) {
    std::vector<int> all_keys;
    for (int i = 0; i < num_params; i++) all_keys.push_back(i);
    std::vector<float> test_params;
    worker->Pull(all_keys, &test_params);
    int count = 0;
    float c_count = 0; //correct count
    while (data_iterator.has_next()) {
        auto& data = data_iterator.next();
        count = count + 1;
        auto& x = data.x;
        float y = data.y;
        if(y < 0) y = 0;
        float pred_y = 0.0;
        for (auto field : x) {
            pred_y += test_params[field.fea] * field.val;
        }
        // pred_y += test_params[num_params - 1];
        pred_y = 1. / (1. + exp(-pred_y));
        pred_y = (pred_y > 0.5) ? 1 : 0;
        if (int(pred_y) == int(y)) { c_count += 1;}
    }
    return c_count/count;
}

}  // namespace
}  // namespace husky
