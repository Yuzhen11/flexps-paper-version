
#include "worker/engine.hpp"
#include "ml/ml.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Start the kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kConsistency, husky::constants::kBSP},
        {husky::constants::kNumWorkers, "4"}
    };
    int kv = kvstore::KVStore::Get().CreateKVStore<float>(hint, 10, 10);  // for bsp server

    auto task = TaskFactory::Get().CreateTask<AutoParallelismTask>();
    task.set_dimensions(10);
    task.set_kvstore(kv);
    task.set_hint(hint);
    task.set_epoch_iter({100, 100});
    task.set_epoch_lambda([](const Info& info, int num_iters) {
        if (info.get_cluster_id() == 0)
            husky::LOG_I << "num_iters: " << num_iters;
    });
    engine.AddTask(task, [](const Info& info) { 
        // dummy lambda. if we cannot remove it, we just leave it here
    });

    engine.Submit();
    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
