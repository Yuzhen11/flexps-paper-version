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

using namespace husky;
using husky::lib::ml::LabeledPointHObj;
using husky::lib::Objective;
using husky::lib::SGDOptimizer;
using husky::lib::SVMObjective;
using husky::lib::SigmoidObjective;
using husky::lib::LassoObjective;

int main(int argc, char** argv) {
    // Get configs
    bool rt = init_with_args(
        argc, argv,
        {"worker_port", "cluster_manager_host", "cluster_manager_port", "hdfs_namenode", "hdfs_namenode_port", "input",
         "num_load_workers", "num_train_workers", 
         "num_features", "trainer", "alpha",
         "batchsize", "num_iters", "lr_coeff", 
         "report_interval", "lambda", "lines_read_per_thread",
         "param_type"});

    ASSERT_MSG(rt, "cannot initialize with args");

    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    int num_train_workers = std::stoi(Context::get_param("num_train_workers"));
    int batchsize = std::stoi(Context::get_param("batchsize"));
    int num_iters = std::stoi(Context::get_param("num_iters"));
    float alpha = std::stof(Context::get_param("alpha"));
    float lr_coeff = std::stof(Context::get_param("lr_coeff"));
    const std::string& trainer = Context::get_param("trainer");
    int num_features = std::stoi(Context::get_param("num_features"));
    int num_params = num_features + 1;  // bias
    int report_interval = std::stoi(Context::get_param("report_interval"));
    float lambda = (Context::get_param("lambda") == "") ? 0. : std::stod(Context::get_param("lambda"));
    int lines_read_per_thread = std::stoi(Context::get_param("lines_read_per_thread"));
    const std::string& param_type = Context::get_param("param_type");
    // Show Config
    if (Context::get_worker_info().get_process_id() == 0) {
        std::stringstream ss;
        ss << "num_load_workers: " << num_load_workers;
        ss << "num_train_workers: " << num_train_workers;
        ss << "batchsize: " << batchsize << std::endl;
        ss << "num_iters: " << num_iters << std::endl;
        ss << "alpha: " << alpha << std::endl;
        ss << "lr_coeff: " << lr_coeff << std::endl;
        ss << "trainer: " << trainer << std::endl;
        ss << "num_features: " << num_features << std::endl;
        ss << "report_interval: " << report_interval << std::endl;
        ss << "lambda: " << lambda << std::endl;
        ss << "lines_read_per_thread: " << lines_read_per_thread << std::endl;
        ss << "param_type: " << param_type << std::endl;
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
    load_task.set_num_workers(num_load_workers);
    engine.AddTask(load_task, [&data_store, num_features, lines_read_per_thread](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, local_id, lines_read_per_thread);
    });

    // 3. Submit load task
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_process_id() == 0)
        husky::LOG_I << YELLOW("Load time: ") << std::to_string(load_time) << " ms";

    // 4. Train task
    int kv = kvstore::KVStore::Get().CreateKVStore<float>("default_assign_vector", -1, -1, num_params);
    TableInfo table_info{
        kv, num_params, 
        husky::ModeType::Hogwild, husky::Consistency::None, 
        husky::WorkerType::None, husky::ParamType::IntegralType};
    if (param_type == "integral") {
        table_info.param_type = husky::ParamType::IntegralType;
    } else if (param_type == "chunk") {
        table_info.param_type = husky::ParamType::ChunkType;
    } else {
        husky::LOG_I << "Unknown param_type: " << param_type;
        return 1;
    }

    auto train_task = TaskFactory::Get().CreateTask<Task>();
    train_task.set_local();
    train_task.set_num_workers(num_train_workers);
    engine.AddTask(train_task, [table_info, trainer, num_params, report_interval, &data_store, lambda, batchsize,
                                num_iters, alpha, lr_coeff, num_train_workers](const Info& info) {
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

        // Config for optimizer
        lib::OptimizerConfig conf;
        conf.num_iters = num_iters;
        // get batch size and learning rate for each worker thread
        conf.batch_size = batchsize / num_train_workers;
        if (info.get_cluster_id() < batchsize % num_train_workers)
            conf.batch_size += 1;
        conf.alpha = alpha / num_train_workers;
        conf.learning_rate_decay = lr_coeff;
        if (info.get_cluster_id() == 0) {
            husky::LOG_I << "Stage begins: { iters:" << conf.num_iters 
              << ", batchsize:" << conf.batch_size 
              << ", alpha:" << conf.alpha 
              << ", lr_coeff:" << conf.learning_rate_decay
              << "}";
        }
        sgd.train(info, table_info, data_store, conf);
        auto end_time = std::chrono::steady_clock::now();
        auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (info.get_cluster_id() == 0) {
            husky::LOG_I << "Stage traintime:" << train_time <<" ms";
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
