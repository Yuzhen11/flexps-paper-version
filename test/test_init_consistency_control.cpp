#include "kvstore/kvstore.hpp"
#include "worker/engine.hpp"
#include "ml/ml.hpp"

using namespace husky;

/*
 * A demo to test InitForConsistencyController in kvserver
 */
int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();

    std::map<std::string, std::string> hint = {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kConsistency, husky::constants::kSSP},
        {husky::constants::kStaleness, "2"},
        {husky::constants::kNumWorkers, "1"},  // may be override
    };
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());
    int kv = kvstore::KVStore::Get().CreateKVStore<float>(hint);

    auto task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>(5, 1);  // stages: 5, num_workers: 1 is useless
    task.set_worker_num({1, 2, 3, 2, 1});
    task.set_worker_num_type({"threads_per_cluster", "threads_per_cluster", "threads_per_cluster", "threads_per_cluster", "threads_per_cluster"});
    task.set_hint(hint);
    task.set_kvstore(kv);
    engine.AddTask(task, [](const Info& info) {
        husky::LOG_I << "Running stage: " << info.get_current_epoch();
        auto worker = ml::CreateMLWorker<float>(info);
        std::vector<float> v;
        worker->Pull({1,2,3}, &v);
        if (info.get_cluster_id() == 0) {
            husky::LOG_I << "pull_res: " << v[0] << " " << v[1] << " " << v[2];
        }
        worker->Push({1,2,3}, {0.1, 0.1, 0.1});
    });
    engine.Submit();

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
