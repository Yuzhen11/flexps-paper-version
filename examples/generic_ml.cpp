#include <vector>

#include "ml/common/mlworker.hpp"
#include "worker/engine.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port"});
    if (!rt)
        return 1;

    Engine engine;
    // Start the kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());
    // Didn't specify the epoch num and thread num, leave master to decide them
    int kv0 = kvstore::KVStore::Get().CreateKVStore<float>();
    auto task1 = TaskFactory::Get().create_task(Task::Type::GenericMLTaskType);
    static_cast<MLTask*>(task1.get())->set_dimensions(10);
    static_cast<MLTask*>(task1.get())->set_kvstore(kv0);
    static_cast<GenericMLTask*>(task1.get())->set_running_type(Task::Type::HogwildTaskType);
    task1->set_total_epoch(2);
    engine.add_task(std::move(task1), [](const Info& info) {
        auto& worker = info.get_mlworker();
        int k = 3;
        worker->Put(k, 0.456);
        float v = worker->Get(k);
        base::log_msg("k: " + std::to_string(k) + " v: " + std::to_string(v));
    });

    engine.submit();
    engine.exit();
    // Stop the kvstore, should stop before mailbox is down
    kvstore::KVStore::Get().Stop();
}
