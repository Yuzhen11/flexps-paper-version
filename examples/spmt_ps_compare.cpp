/*
 * Compare SPMT and PS (BSP/SSP)
 * Use ConfigurableWorkersTask to run instances on the same process
 * Compare Hogwild! and PS (ASP) model
 *
 * Config Example
 *
 * task_scheduler_type=greedy
 * num_load_workers=4
 * num_train_workers=4
 *
 * input=hdfs:///ml/a9
 * alpha=0.5
 * num_iters=-1
 * num_features=123
 * train_epoch=1
 */

#include <chrono>
#include <vector>

#include "core/color.hpp"
#include "datastore/datastore_utils.hpp"
#include "lib/app_config.hpp"
#include "lib/load_data.hpp"
#include "lib/ml_lambda.hpp"
#include "lib/task_utils.hpp"
#include "worker/engine.hpp"

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

int main(int argc, char** argv) {
    // Set config
    config::InitContext(argc, argv);
    auto config = config::SetAppConfigWithContext();
    if (Context::get_worker_info().get_process_id() == 0) config::ShowConfig(config);

    // Hint for SPMT/Hogwild!
    std::map<std::string, std::string> hint_spmt = {
        {husky::constants::kConsistency, config.kConsistency},
        {husky::constants::kNumWorkers, std::to_string(config.num_train_workers)},
        {husky::constants::kStaleness, std::to_string(config.staleness)}
    };

    // Hint for PS
    std::map<std::string, std::string> hint_ps = {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kConsistency, config.kConsistency},
        {husky::constants::kNumWorkers, std::to_string(config.num_train_workers)},
        {husky::constants::kStaleness, std::to_string(config.staleness)}
    };

    if (config.use_chunk) {
        hint_spmt.insert({husky::constants::kParamType, husky::constants::kChunkType});
        hint_ps.insert({husky::constants::kParamType, husky::constants::kChunkType});
    }
    if (config.use_direct_model_transfer) {
        hint_spmt.insert({husky::constants::kEnableDirectModelTransfer, "on"});
    }
    if (config.use_chunk && config.use_direct_model_transfer) {
        assert(false);
    }

    if (config.kConsistency == husky::constants::kASP) {
        hint_spmt.insert({husky::constants::kType, husky::constants::kHogwild});
    } else {
        hint_spmt.insert({husky::constants::kType, husky::constants::kSPMT});
        if (config.kConsistency == husky::constants::kSSP) {
            const std::vector<std::string> ps_worker_types{"PSWorker", "SSPWorker", "SSPWorkerChunk", "PSSharedWorker", "PSSharedChunkWorker"};
            assert(std::find(ps_worker_types.begin(), ps_worker_types.end(), config.ps_worker_type) != ps_worker_types.end());
            hint_ps.insert({husky::constants::kWorkerType, config.ps_worker_type});
        }
    }

    auto& engine = Engine::Get();
    // Start KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(), Context::get_zmq_context());

    // Create DataStore
    datastore::DataStore<LabeledPointHObj<float, float, true>> data_store(Context::get_worker_info().get_num_local_workers());

    // Load task
    auto load_task = TaskFactory::Get().CreateTask<HuskyTask>(1, config.num_load_workers);
    engine.AddTask(load_task, [&data_store, config](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, config.num_features, local_id);
    });
    engine.Submit();

    // Train task with SPMT
    auto train_task_spmt = TaskFactory::Get().CreateTask<MLTask>();
    train_task_spmt.set_num_workers(config.num_train_workers);
    train_task_spmt.set_total_epoch(config.train_epoch);
    train_task_spmt.set_dimensions(config.num_params);
    // Train task with PS
    auto train_task_ps = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>(config.train_epoch, config.num_train_workers);
    train_task_ps.set_dimensions(config.num_params);
    train_task_ps.set_worker_num({config.num_train_workers});
    train_task_ps.set_worker_num_type({"local_threads"});
    auto train_task_lambda = [&data_store, config](const Info& info) {
        lambda::train(data_store, config, info);
    };

    // Create KVStore and set hint
    int kv_spmt = create_kvstore_and_set_hint(hint_spmt, train_task_spmt, config.num_params);
    int kv_ps = create_kvstore_and_set_hint(hint_ps, train_task_ps, config.num_params);

    // Submit SPMT task
    engine.AddTask(train_task_spmt, train_task_lambda);
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto spmt_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    // Submit PS task
    engine.AddTask(train_task_ps, train_task_lambda);
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto ps_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    if (Context::get_worker_info().get_process_id() == 0) {
        husky::LOG_I << YELLOW("spmt train time: " + std::to_string(spmt_time) + " ms");
        husky::LOG_I << YELLOW("ps train time: " + std::to_string(ps_time) + " ms");
    }

    engine.Exit();
    kvstore::KVStore::Get().Stop();
    return 0;
}
