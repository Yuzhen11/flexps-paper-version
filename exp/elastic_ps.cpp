#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <cassert>
#include <chrono>
#include <string>
#include <vector>

#include "core/task.hpp"
#include "datastore/datastore.hpp"
#include "datastore/datastore_utils.hpp"
#include "lib/app_config.hpp"
#include "lib/load_data.hpp"
#include "lib/ml_lambda.hpp"
#include "lib/objectives.hpp"
#include "lib/optimizers.hpp"
#include "lib/task_utils.hpp"
#include "worker/engine.hpp"

#include "husky/lib/ml/feature_label.hpp"

/*
 * Configuration
 *
 * batch_sizes=<stage_0_size>,<stage_1_size>...
 * nums_iters=<stage_i_iters>...
 * alphas=<stage_i_learning_rate>...
 * lr_coeffs=<stage_i_learning_rate_decay_steps>...
 * nums_train_workers=<stage_i_num_train_workers>...
 * num_load_workers=<number>
 * report_iterval=0 # no report
 * trainer=[lr|svm|lasso]
 * lambda=<float for svm or lasso>
 * kType=PS
 */

using namespace husky;
using husky::lib::ml::LabeledPointHObj;
using husky::lib::Objective;
using husky::lib::SGDOptimizer;
using husky::lib::SVMObjective;
using husky::lib::SigmoidObjective;
using husky::lib::LassoObjective;

template <typename T>
void vec_to_str(const std::string& name, std::vector<T>& vec, std::stringstream& ss) {
    ss << name;
    for (auto& v : vec)
        ss << "," << v;
    ss << "\n";
}

template <typename T>
void get_stage_conf(const std::string& conf_str, std::vector<T>& vec, int num_stage) {
    std::vector<std::string> split_result;
    boost::split(split_result, conf_str, boost::is_any_of(","), boost::algorithm::token_compress_on);
    vec.reserve(num_stage);
    for (auto& i : split_result) {
        vec.push_back(std::stoi(i));
    }
    assert(vec.size() == num_stage);
}

int main(int argc, char** argv) {
    // Get configs
    config::InitContext(argc, argv, {"batch_sizes", "nums_iters", "alphas", "lr_coeffs", "nums_train_workers",
                                     "num_load_workers", "report_interval" /*, "model_input"*/, "lambda"});
    config::AppConfig config = config::SetAppConfigWithContext();
    auto batch_size_str = Context::get_param("batch_sizes");
    auto nums_workers_str = Context::get_param("nums_train_workers");
    auto nums_iters_str = Context::get_param("nums_iters");
    auto alphas_str = Context::get_param("alphas");
    auto lr_coeffs_str = Context::get_param("lr_coeffs");
    auto model_file = Context::get_param("model_input");
    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    int report_interval = std::stoi(Context::get_param("report_interval"));
    float lambda = (Context::get_param("lambda") == "") ? 0. : std::stod(Context::get_param("lambda"));
    // Get configs for each stage
    std::vector<int> batch_sizes;
    std::vector<int> nums_workers;
    std::vector<int> nums_iters;
    std::vector<float> alphas;
    std::vector<float> lr_coeffs;
    get_stage_conf(batch_size_str, batch_sizes, config.train_epoch);
    get_stage_conf(nums_workers_str, nums_workers, config.train_epoch);
    get_stage_conf(nums_iters_str, nums_iters, config.train_epoch);
    get_stage_conf(alphas_str, alphas, config.train_epoch);
    get_stage_conf(lr_coeffs_str, lr_coeffs, config.train_epoch);

    // Show Config
    if (Context::get_worker_info().get_process_id() == 0) {
        config::ShowConfig(config);
        std::stringstream ss;
        vec_to_str("batch_sizes", batch_sizes, ss);
        vec_to_str("nums_workers", nums_workers, ss);
        vec_to_str("nums_iters", nums_iters, ss);
        vec_to_str("alphas", alphas, ss);
        vec_to_str("lr_coeffs", lr_coeffs, ss);
        husky::LOG_I << ss.str();
    }

    auto& engine = Engine::Get();
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    // Load Data
    // 1. Create DataStore
    datastore::DataStore<LabeledPointHObj<float, float, true>> data_store(
        Context::get_worker_info().get_num_local_workers());
    // 2. Add load task
    std::vector<float> benchmark_model;
    int kv_true = -1;
    if (model_file != "") {
        kv_true = kvstore::KVStore::Get().CreateKVStore<float>();
        assert(kv_true != -1);
    }
    auto load_task = TaskFactory::Get().CreateTask<HuskyTask>(1, num_load_workers);
    engine.AddTask(load_task, [&data_store, config, &benchmark_model, &model_file, &kv_true](const Info& info) {
        /*
        if (model_file != "" && info.get_cluster_id() == 0) {
            lr::load_benchmark_model(benchmark_model, model_file, config.num_params);
            std::vector<husky::constants::Key> keys(config.num_params);
            for (int i = 0; i < config.num_params; ++i) keys[i] = i;
            auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
            kvworker->Wait(kv_true, kvworker->Push(kv_true, keys, benchmark_model));
        }
        */
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, config.num_features, local_id);
    });

    // 3. Submit load task
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_process_id() == 0)
        husky::LOG_I << YELLOW("Load time: ") << std::to_string(load_time) << " ms";

    // 4. Train task
    auto train_task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>();
    train_task.set_dimensions(config.num_params);
    train_task.set_total_epoch(config.train_epoch);
    train_task.set_num_workers(config.num_train_workers);
    train_task.set_worker_num(nums_workers);
    train_task.set_worker_num_type(std::vector<std::string>(nums_workers.size(), "threads_per_cluster"));
    auto hint = config::ExtractHint(config);
    int kv = create_kvstore_and_set_hint(hint, train_task, config.num_params);

    engine.AddTask(train_task, [&report_interval, &data_store, &config, lambda, &batch_sizes, &nums_iters, &alphas,
                                &lr_coeffs, &nums_workers, &benchmark_model, kv_true](const Info& info) {
        // set objective
        Objective* objective_ptr;
        if (config.trainer == "lr") {
            objective_ptr = new SigmoidObjective(config.num_params);
        } else if (config.trainer == "lasso") {
            objective_ptr = new LassoObjective(config.num_params, lambda);
        } else {  // default svm
            objective_ptr = new SVMObjective(config.num_params, lambda);
        }
        // set optimizer
        SGDOptimizer sgd(objective_ptr, report_interval);

        int current_stage = info.get_current_epoch();
        config::AppConfig conf = config;
        conf.num_train_workers = nums_workers[current_stage];
        husky::ASSERT_MSG(conf.num_train_workers > 0, "num_train_workers must be positive!");
        conf.num_iters = nums_iters[current_stage];
        // get batch size and learning rate for each worker thread
        conf.batch_size = batch_sizes[current_stage] / conf.num_train_workers;
        if (info.get_cluster_id() < batch_sizes[current_stage] % conf.num_train_workers)
            conf.batch_size += 1;
        conf.alpha = alphas[current_stage] / conf.num_train_workers;
        conf.learning_rate_coefficient = lr_coeffs[current_stage];
        if (info.get_cluster_id() == 0) {
            husky::LOG_I << "Stage " << current_stage << ": " << conf.num_iters << "," << conf.batch_size << ","
                         << conf.alpha << "," << conf.learning_rate_coefficient;
            /* load benchmark model
            if (kv_true != -1 && benchmark_model.empty()) {
                std::vector<husky::constants::Key> keys(conf.num_params);
                for (int i = 0; i < conf.num_params; ++i) keys[i] = i;
                auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
                kvworker->Wait(kv_true, kvworker->Pull(kv_true, keys, &benchmark_model));
            }
            */
        }
        int accum_iter = 0;
        for (int i = 0; i < info.get_current_epoch(); ++i) {
            accum_iter += nums_iters[i];
        }
        sgd.train(info, data_store, conf, accum_iter);
        // lr::sgd_train(info, data_store, conf, benchmark_model, report_interval, accum_iter, false);
    });

    // 5. Submit train task
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_process_id() == 0) {
        husky::LOG_I << "Train time: " << train_time;
    }

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
