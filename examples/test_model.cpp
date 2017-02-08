#include <chrono>
#include <vector>

#include "ml/common/mlworker.hpp"
#include "worker/engine.hpp"

#include "core/color.hpp"

using namespace husky;

std::vector<float> answer(10, 1000.0);
void get_keys(std::vector<husky::constants::Key>& keys, const Info& info) {
    if (info.get_cluster_id() == 0) {
        keys = {0,1,2,3};
    } else if (info.get_cluster_id() == 1) {
        keys = {3,4,5,8};
    } else {
        keys = {6,7,9};
    }
}

bool check(std::vector<husky::constants::Key>& keys, std::vector<float>& rets) {
    for (int i = 0; i < keys.size(); ++i) {
        if (rets[i] != answer[keys[i]]) return false;
    }
    return true;
}

auto test = [](const Info& info) {
    auto& worker = info.get_mlworker();
    int num_iter = 1001;
    std::vector<float> rets;
    std::vector<husky::constants::Key> keys;
    get_keys(keys, info);
    for (int i = 0; i < num_iter; ++i) {
        worker->Pull(keys, &rets);
        /*
        if (i % 100 == 0) {
            husky::LOG_I << BLUE("id:" + std::to_string(info.get_local_id()) + ", " + "iter: " + std::to_string(i) + ", " + std::to_string(keys[0]) + ": " + std::to_string(rets[0]));
        }
        */

        std::vector<float> delta(keys.size(), 1.0);
        worker->Push(keys, delta);
    }
    assert(check(keys, rets));
};

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    answer[3] = 2000.0;
    auto& engine = Engine::Get();
    // Start the kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    int kv = kvstore::KVStore::Get().CreateKVStore<float>();
    auto task = TaskFactory::Get().CreateTask<MLTask>();
    task.set_dimensions(10);
    task.set_kvstore(kv);
    task.set_hint("SPMT:BSP");  // set the running type explicitly
    task.set_num_workers(3);
    engine.AddTask(task, [](const Info& info) {
        test(info);
    });

    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    husky::LOG_I << CLAY("elapsed time: " + std::to_string(elapsed) + " ms");
    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
