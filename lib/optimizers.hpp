#pragma once

#include "core/constants.hpp"
#include "core/info.hpp"
#include "datastore/datastore.hpp"
#include "datastore/datastore_utils.hpp"
#include "lib/objectives.hpp"
#include "lib/utils.hpp"
#include "ml/ml.hpp"
#include "ml/mlworker/mlworker.hpp"

#include "husky/lib/ml/feature_label.hpp"
#include "husky/lib/vector.hpp"

namespace husky {
namespace lib {
namespace {

struct OptimizerConfig {
    int num_iters = 10;
    float alpha = 0.1;
    int batch_size = 10;
    int learning_rate_decay = 10;
};

class Optimizer {
   public:
    Optimizer(Objective* objective, int report_interval) : objective_(objective), report_interval_(report_interval) {}

    virtual void train(const Info& info, const TableInfo& table_info,
                       datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store,
                       const OptimizerConfig& config, int iter_offset = 0) = 0;

   protected:
    Objective* objective_;
    int report_interval_ = 0;
};

class SGDOptimizer : Optimizer {
   public:
    SGDOptimizer(Objective* objective, int report_interval) : Optimizer(objective, report_interval) {}

    void train(const Info& info, const TableInfo& table_info,
               datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store, const OptimizerConfig& config,
               int iter_offset = 0) override {
        // 1. Get worker for communication with server
        auto worker = ::ml::CreateMLWorker<float>(info, table_info);

        // 2. Create BatchDataSampler for mini-batch SGD
        datastore::BatchDataSampler<husky::lib::ml::LabeledPointHObj<float, float, true>> batch_data_sampler(
            data_store, config.batch_size);
        batch_data_sampler.random_start_point();

        // 3. Main loop
        Timer train_timer(true);
        for (int iter = iter_offset; iter < config.num_iters + iter_offset; ++iter) {
            // a. Train
            float alpha = config.alpha / (iter / config.learning_rate_decay + 1);
            alpha = std::max(1e-5f, alpha);
            update(worker, batch_data_sampler, alpha);

            // b. Report loss on training samples
            if (report_interval_ != 0 && (iter + 1) % report_interval_ == 0) {
                train_timer.pause();
                std::vector<float> vals;
                if (info.get_cluster_id() == 0) {  // let the cluster leader do the report
                    // pull model
                    std::vector<husky::constants::Key> keys;
                    objective_->all_keys(&keys);
                    worker->Pull(keys, &vals);
                    worker->Push({keys[0]}, {0});
                    // test with training samples
                    auto loss = objective_->get_loss(data_store, vals);
                    husky::LOG_I << "Task " << info.get_task_id() << ": Iter, Time, Loss: " << iter << ","
                                 << train_timer.elapsed_time() << "," << std::setprecision(15) << loss;
                } else {
                    worker->Pull({0}, &vals);
                    worker->Push({0}, {0});
                }
                train_timer.start();
            }
        }
    }

   private:
    void update(const std::unique_ptr<::ml::mlworker::GenericMLWorker<float>>& worker,
                datastore::BatchDataSampler<LabeledPointHObj<float, float, true>>& batch_data_sampler, float alpha) {
        // 1. Prepare all the parameter keys in the batch
        std::vector<husky::constants::Key> keys = batch_data_sampler.prepare_next_batch();
        objective_->process_keys(&keys);
        std::vector<float> params, delta;
        delta.resize(keys.size(), 0.0);

        // 2. Pull parameters
        worker->Pull(keys, &params);

        // 3. Calculate gradients
        objective_->get_gradient(batch_data_sampler.get_data_ptrs(), keys, params, &delta);

        // 4. Adjust step size
        for (auto& d : delta) {
            d *= -alpha;
        }

        // 5. Push updates
        worker->Push(keys, delta);
    }
};

}  // namespace anonymous
}  // namespace lib
}  // namespace husky
