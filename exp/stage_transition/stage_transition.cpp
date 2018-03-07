#include <vector>

#include "worker/engine.hpp"
#include "ml/ml.hpp"

#include "core/color.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port",
            "threads_per_process"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Start the kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    int threads_per_process = std::stoi(Context::get_param("threads_per_process"));
    if (threads_per_process < 0 || threads_per_process > 20) {
        husky::LOG_I << "threads_per_process error: " << threads_per_process;
        return 1;
    }
    if (Context::get_process_id() == 0)
        husky::LOG_I << "Using " << threads_per_process << " workers";

    const int dims = 10;
    const int kv = kvstore::KVStore::Get().CreateKVStore<float>("bsp_add_vector", -1, -1, dims, 10);
    auto task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>();
    task.set_total_epoch(2);
    task.set_worker_num({threads_per_process, threads_per_process});
    task.set_worker_num_type({"threads_per_worker", "threads_per_worker"});
    TableInfo table_info {
        kv, dims,
        husky::ModeType::PS,
        husky::Consistency::BSP,
        husky::WorkerType::PSWorker
    };
    
    engine.AddTask(task, [table_info](const Info& info) {
        if (info.get_cluster_id() == 0) 
            husky::LOG_I << "PS BSP Model running";
        auto worker = ml::CreateMLWorker<float>(info, table_info);
    });

    engine.Submit();
    engine.Exit();
    // Stop the kvstore, should stop before mailbox is down
    kvstore::KVStore::Get().Stop();
}
