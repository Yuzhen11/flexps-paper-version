#pragma once

#include "core/info.hpp"

#include "lib/app_config.hpp"
#include "datastore/datastore.hpp"
#include "datastore/datastore_utils.hpp"

#include "lib/lr_updater.hpp"
#include "lib/svm_updater.hpp"

namespace husky {
namespace lambda {
namespace {

auto train = [](datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store,
                config::AppConfig config,
                const Info& info) {
    // Create a DataStoreWrapper
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
    for (int iter = 0; iter < config.num_iters; ++iter) {
        if (config.trainer == "lr") {
            // sgd_update(worker, data_sampler, config.alpha);
            lr::batch_sgd_update_lr(worker, batch_data_sampler, config.alpha);
        } else if (config.trainer == "svm") {
            svm::batch_sgd_update_svm_dense(worker, batch_data_sampler, config.alpha, config.num_params);
        }

        if (iter % 10 == 0) {
            // Testing, now all the threads need to run `get_test_error`, it is for PS.
            // So it won't mess up the iteration
            datastore::DataIterator<LabeledPointHObj<float, float, true>> data_iterator(data_store);
            float test_error = -1;
            if (config.trainer == "lr") {
                test_error = lr::get_test_error_lr_v2(worker, data_iterator, config.num_params);
            } else if (config.trainer == "svm") {
                test_error = svm::get_test_error_svm_v2(worker, data_iterator, config.num_params);
            }
            if (info.get_cluster_id() == 0) {
                husky::LOG_I << "Iter:" << std::to_string(iter) << " Accuracy is " << test_error;
            }
        }
    }
};

}  // namespace anonymous
}  // namespace lambda 
}  // namespace husky
