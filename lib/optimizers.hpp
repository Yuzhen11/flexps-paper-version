#pragma once

#include "core/constants.hpp"
#include "core/info.hpp"
#include "core/table_info.hpp"
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

    void trainChunkModel(const Info& info, const TableInfo& table_info,
               datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store, const OptimizerConfig& config,
               int chunk_size, int iter_offset = 0) { 
        if (table_info.worker_type != husky::WorkerType::PSNoneChunkWorker) {
            husky::LOG_I<<"Please set WorkerType to PSNoneChunkWorker";
            return;
        }

        // 1. Get worker for communication with server
        auto worker = ::ml::CreateMLWorker<float>(info, table_info);

        // 2. Create BatchDataSampler for mini-batch SGD
        datastore::BatchDataSampler<husky::lib::ml::LabeledPointHObj<float, float, true>> batch_data_sampler(
            data_store, config.batch_size);
        batch_data_sampler.random_start_point();

        // 3. Main loop
        for (int iter = iter_offset; iter < config.num_iters + iter_offset; ++iter) {
            float alpha = config.alpha / (iter / config.learning_rate_decay + 1);
            alpha = std::max(1e-5f, alpha);
            updateDenseModel(worker, batch_data_sampler, alpha, chunk_size, info);
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

    void updateDenseModel(const std::unique_ptr<::ml::mlworker::GenericMLWorker<float>>& worker,
                datastore::BatchDataSampler<LabeledPointHObj<float, float, true>>& batch_data_sampler,
                float alpha, int chunk_size, const Info& info) {

        batch_data_sampler.prepare_next_batch_data();
        // 0. init keys
        std::vector<husky::constants::Key> keys;
        objective_->all_keys(&keys);

        // 1. init chunk keys and push pull pointers
        int num_chunks = ((int)keys.size() + chunk_size - 1) / chunk_size;
        std::vector<husky::constants::Key> chunk_keys(num_chunks);
        std::iota(chunk_keys.begin(), chunk_keys.end(), 0);
        std::vector<std::vector<float>*> pull_ptrs(num_chunks);
        std::vector<std::vector<float>*> push_ptrs(num_chunks);

        std::vector<std::vector<float>> params_inchunks(num_chunks);
        std::vector<std::vector<float>> delta_inchunks(num_chunks);

        for (int i = 0; i < num_chunks; ++i) {
            pull_ptrs[i] = &params_inchunks[i];
            push_ptrs[i] = &delta_inchunks[i];
        }

        // 2. Pull parameters
        auto t1 = std::chrono::steady_clock::now();
        worker->PullChunks(chunk_keys, pull_ptrs);
        auto t2 = std::chrono::steady_clock::now();

        std::vector<float> params;
        flattenTo(params_inchunks, params);
        auto tf = std::chrono::steady_clock::now();

        std::vector<float> delta;
        delta.resize(keys.size(), 0.0);
        // 3. Calculate gradients
        objective_->get_gradient(batch_data_sampler.get_data_ptrs(), keys, params, &delta);

        // 4. Adjust step size
        for (auto& d : delta) {
            d *= -alpha;
        }

        auto tc = std::chrono::steady_clock::now();
        compressTo(delta, delta_inchunks, chunk_size);
        // 5. Push updates
        auto t3 = std::chrono::steady_clock::now();
        worker->PushChunks(chunk_keys, push_ptrs);
        auto t4 = std::chrono::steady_clock::now();
        if (info.get_cluster_id() == 0) {
            auto pull_time = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
            auto push_time = std::chrono::duration_cast<std::chrono::microseconds>(t4-t3);
            auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t3 -t2);
            auto flatten_time = std::chrono::duration_cast<std::chrono::microseconds>(tf -t2);
            auto compress_time = std::chrono::duration_cast<std::chrono::microseconds>(t3 -tc);
            husky::LOG_I << "compute: " << compute_time.count()/1000.0 << " ms"
            << ", pull:" << pull_time.count()/1000.0 << " ms"
            << ", push:" << push_time.count()/1000.0 << " ms"
            << ", flatten:" << flatten_time.count()/1000.0 << " ms"
            << ", conpress:" << compress_time.count()/1000.0 << " ms";
        }

    }

    void flattenTo(const std::vector<std::vector<float>>& left, std::vector<float>& right) {
        for (int i=0; i<left.size(); i++) {
            right.insert(right.end(), left[i].begin(), left[i].end());
        }
    } 

    void compressTo(const std::vector<float>& left, std::vector<std::vector<float>>& right, int chunk_size) {
        right.resize(left.size()/chunk_size);
        for (int i=0; i< left.size(); i++) {
            int r = i/chunk_size;
            int c = i%chunk_size;
            right[r].resize(chunk_size);
            right[r][c] = left[i];
        }
    } 
};

}  // namespace anonymous
}  // namespace lib
}  // namespace husky
