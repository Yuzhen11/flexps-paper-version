#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <cassert>
#include <chrono>
#include <string>
#include <vector>

#include "core/task.hpp"
#include "datastore/datastore.hpp"
#include "datastore/datastore_utils.hpp"
#include "lib/load_data.hpp"
#include "lib/objectives.hpp"
#include "lib/optimizers.hpp"
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
    bool rt = init_with_args(
        argc, argv,
        {"worker_port", "cluster_manager_host", "cluster_manager_port", "hdfs_namenode", "hdfs_namenode_port", "input",
         "num_features", "train_epoch", "trainer", "staleness", "consistency", "batch_sizes", "nums_iters", "alphas",
         "nums_train_workers", "lr_coeffs", "num_load_workers", "report_interval", "lambda", "chunk_size"});
    ASSERT_MSG(rt, "cannot initialize with args");

    auto batch_size_str = Context::get_param("batch_sizes");
    auto nums_workers_str = Context::get_param("nums_train_workers");
    auto nums_iters_str = Context::get_param("nums_iters");
    auto alphas_str = Context::get_param("alphas");
    auto lr_coeffs_str = Context::get_param("lr_coeffs");

    int staleness = std::stoi(Context::get_param("staleness"));
    int chunk_size = std::stoi(Context::get_param("chunk_size"));
    int train_epoch = std::stoi(Context::get_param("train_epoch"));
    const std::string& trainer = Context::get_param("trainer");

    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    int num_features = std::stoi(Context::get_param("num_features"));
    int num_params = num_features + 1;  // bias
    int report_interval = std::stoi(Context::get_param("report_interval"));
    float lambda = (Context::get_param("lambda") == "") ? 0. : std::stod(Context::get_param("lambda"));
    // Get configs for each stage
    std::vector<int> batch_sizes;
    std::vector<int> nums_workers;
    std::vector<int> nums_iters;
    std::vector<float> alphas;
    std::vector<float> lr_coeffs;
    get_stage_conf(batch_size_str, batch_sizes, train_epoch);
    get_stage_conf(nums_workers_str, nums_workers, train_epoch);
    get_stage_conf(nums_iters_str, nums_iters, train_epoch);
    get_stage_conf(alphas_str, alphas, train_epoch);
    get_stage_conf(lr_coeffs_str, lr_coeffs, train_epoch);

    // Show Config
    if (Context::get_worker_info().get_process_id() == 0) {
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
    auto load_task = TaskFactory::Get().CreateTask<Task>();
    load_task.set_total_epoch(1);
    load_task.set_num_workers(num_load_workers);
    engine.AddTask(load_task, [&data_store, num_features](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, local_id);
    });

    // 3. Submit load task
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_process_id() == 0)
        husky::LOG_I << YELLOW("Load time: ") << std::to_string(load_time) << " ms";

    // 4. Train task
    std::string storage_type = "bsp_add_vector";
    husky::Consistency consistency = husky::Consistency::BSP;
    if (Context::get_param("consistency") == "SSP") {
        consistency = husky::Consistency::SSP;
        storage_type = "ssp_add_vector";
    } else {
        LOG_I<<"Currently not supported!";
        return -1;
    } 

    int kv = kvstore::KVStore::Get().CreateKVStore<float>(storage_type, 1, staleness, num_params, chunk_size);
    TableInfo table_info{
        kv, num_params,
        husky::ModeType::PS,
        consistency,
        husky::WorkerType::PSNoneChunkWorker, 
        husky::ParamType::None,
    };
    if (consistency == husky::Consistency::SSP) {
        table_info.kStaleness = staleness;
    }

    auto train_task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>();
    train_task.set_total_epoch(train_epoch);
    train_task.set_worker_num(nums_workers);
    train_task.set_worker_num_type(std::vector<std::string>(nums_workers.size(), "threads_per_worker"));

    engine.AddTask(train_task, [table_info, trainer, num_params, &report_interval, &data_store, lambda, &batch_sizes,
                                &nums_iters, &alphas, &lr_coeffs, &nums_workers, chunk_size](const Info& info) {
    auto start_time = std::chrono::steady_clock::now();
        // set objective
        Objective* objective_ptr;
        if (trainer == "lr") {
            objective_ptr = new SigmoidObjective(num_params);
        } else if (trainer == "lasso") {
            objective_ptr = new LassoObjective(num_params, lambda);
        } else {  // default svm
            objective_ptr = new SVMObjective(num_params, lambda);
        }
        // set optimizer
        SGDOptimizer sgd(objective_ptr, report_interval);

        int current_stage = info.get_current_epoch();
        int num_train_workers = nums_workers[current_stage] * info.get_worker_info().get_num_processes();
        husky::ASSERT_MSG(num_train_workers > 0, "num_train_workers must be positive!");

        // Config for optimizer
        lib::OptimizerConfig conf;
        conf.num_iters = nums_iters[current_stage];
        // get batch size and learning rate for each worker thread
        conf.batch_size = batch_sizes[current_stage] / num_train_workers;
        if (info.get_cluster_id() < batch_sizes[current_stage] % num_train_workers)
            conf.batch_size += 1;
        conf.alpha = alphas[current_stage] / num_train_workers;
        conf.learning_rate_decay = lr_coeffs[current_stage];
        if (info.get_cluster_id() == 0) {
            husky::LOG_I << "Stage " << current_stage << ": " << conf.num_iters << "," << conf.batch_size << ","
                         << conf.alpha << "," << conf.learning_rate_decay;
        }
        int accum_iter = 0;
        for (int i = 0; i < info.get_current_epoch(); ++i) {
            accum_iter += nums_iters[i];
        }
        sgd.trainChunkModel(info, table_info, data_store, conf, chunk_size, accum_iter);
    auto end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (info.get_cluster_id() == 0) {
        husky::LOG_I << "Stage " <<current_stage << " traintime:" << train_time <<"ms";
    }
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
