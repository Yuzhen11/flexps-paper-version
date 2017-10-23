#include "ml/ml.hpp"
#include "worker/engine.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    int kv = kvstore::KVStore::Get().CreateKVStore<float>("default_assign_map", -1, -1, 10, 10);
    auto task = TaskFactory::Get().CreateTask<AutoParallelismTask>();
    // task.set_epoch_iters({100, 100});
    task.set_epoch_iters_and_batchsizes({100, 100}, {500, 600});
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
