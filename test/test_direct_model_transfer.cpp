#include "worker/engine.hpp"
#include "ml/ml.hpp"

#include "core/color.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Start the kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    int num_params = 5;
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>({}, num_params, num_params);
    auto task1 = TaskFactory::Get().CreateTask<MLTask>();
    task1.set_total_epoch(10);
    task1.set_dimensions(num_params);
    task1.set_kvstore(kv1);
    task1.set_hint({{husky::constants::kType, husky::constants::kSingle}});  // set the running type explicitly
    engine.AddTask(task1, [num_params](const Info& info) {
        auto worker = ml::CreateMLWorker<float>(info);
        std::vector<husky::constants::Key> keys(num_params);
        for (size_t i = 0; i < keys.size(); ++ i) keys[i] = i;
        std::vector<float> res;
        worker->Pull(keys, &res);
        for (size_t i = 0; i < res.size(); ++ i) {
            std::cout << res[i] << " ";
        }
        std::cout << std::endl;
        worker->Push({2}, {1});
    });
    engine.Submit();
    engine.Exit();
    // Stop the kvstore, should stop before mailbox is down
    kvstore::KVStore::Get().Stop();
}
