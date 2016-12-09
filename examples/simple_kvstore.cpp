#include "worker/engine.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port"});
    if (!rt) return 1;

    Engine engine;

    auto task = TaskFactory::Get().create_task(Task::Type::BasicTaskType, 1, 1);
    int kv0 = engine.create_kvstore<int>();
    int kv1 = engine.create_kvstore<float>();
    engine.add_task(std::move(task), [kv0, kv1](const Info& info) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
        std::vector<int> keys{0};
        std::vector<float> vals{2.0};
        int ts = kvworker->PushLocal(kv1, info.get_proc_id(), keys, vals);
        // int ts = kvworker->Push(kv1, keys, vals);
        kvworker->Wait(kv1, ts);
        husky::base::log_msg("Push Done!");

        std::vector<float> rets;
        kvworker->Wait(kv1, kvworker->PullLocal(kv1, info.get_proc_id(), keys, &rets));
        // kvworker->Wait(kv1, kvworker->Pull(kv1, keys, &rets));
        base::log_msg(std::to_string(rets[0]));
    });
    engine.submit();
    engine.exit();
}
