#pragma once

#include "datastore/datastore_utils.hpp"
#include "ml/mlworker/mlworker.hpp"

using husky::lib::ml::LabeledPointHObj;

namespace husky {
namespace svm {
namespace {

// The mini-batch SGD updator
void batch_sgd_update_svm_dense(const std::unique_ptr<ml::mlworker::GenericMLWorker>& worker,
                      datastore::BatchDataSampler<LabeledPointHObj<float, float, true>>& batch_data_sampler, 
                      float alpha,
                      int num_params) {
    alpha /= batch_data_sampler.get_batch_size();
    batch_data_sampler.prepare_next_batch();
    // dense!
    std::vector<husky::constants::Key> keys(num_params);
    for (int i = 0; i < num_params; ++ i) {
        keys[i] = i;
    }
    std::vector<float> params;  // dense
    std::vector<float> delta;  // dense
    delta.resize(keys.size(), 0.0);
    worker->Pull(keys, &params);
    for (auto data : batch_data_sampler.get_data_ptrs()) {
        auto& x = data->x;
        float y = data->y;  // +1/-1
        // predict
        float pred_y = 0.0;
        int i = 0;
        for (auto field : x) {
            pred_y += params[field.fea] * field.val;
        }
        // lacking bias
        pred_y *= y;
        if (pred_y < 1) {
            // TODO: How to set the lambda?
            // wi = wi - alpha * (wi - yi*xi*lambda)
            // -> wi = wi - alpha * wi + alpha * xi * yi * lambda
            for (auto field : x) {
                delta[field.fea] += alpha * field.val * y * 2;
            }
        }
        for (auto field : x) {
            delta[field.fea] -= alpha * params[field.fea];
        }
    }
    worker->Push(keys, delta);
}

float get_test_error_svm_v2(const std::unique_ptr<ml::mlworker::GenericMLWorker>& worker,
                        datastore::DataIterator<LabeledPointHObj<float, float, true>> data_iterator, int num_params,
                        int test_samples = -1) {
    test_samples = 1000;
    std::vector<husky::constants::Key> all_keys;
    for (int i = 0; i < num_params; i++)
        all_keys.push_back(i);
    worker->Prepare_v2(all_keys);
    int count = 0;
    float c_count = 0;  // correct count
    while (data_iterator.has_next()) {
        auto& data = data_iterator.next();
        count = count + 1;
        auto& x = data.x;
        float y = data.y;
        float pred_y = 0.0;
        for (auto field : x) {
            pred_y += worker->Get_v2(field.fea) * field.val;
        }
        if (pred_y * y > 0) {
            c_count += 1;
        }
        if (count == test_samples)
            break;
    }
    worker->Clock_v2();
    return c_count / count;
}

}  // namespace
}  // namespace svm
}  // namespace husky
