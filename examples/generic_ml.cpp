#include <vector>

#include "ml/common/mlworker.hpp"
#include "worker/engine.hpp"

#include "core/color.hpp"
#include "kvstore/consistency/bsp_server.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Start the kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());
    // Didn't specify the epoch num and thread num, leave cluster_manager to decide them
    
    //  A Hogwild! Task
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
    auto task1 = TaskFactory::Get().CreateTask<GenericMLTask>();
    task1.set_dimensions(10);
    task1.set_kvstore(kv1);
    task1.set_running_type(Task::Type::HogwildTaskType);  // set the running type explicitly
    task1.set_total_epoch(2);  // 2 epochs
    task1.set_num_workers(4);  // 4 workers
    engine.AddTask(task1, [](const Info& info) {
        auto& worker = info.get_mlworker();
        // int k = 3;
        // worker->Put(k, 0.456);
        // float v = worker->Get(k);
        // base::log_msg("k: " + std::to_string(k) + " v: " + std::to_string(v));
        int start = info.get_cluster_id();
        for (int i = 0; i < 10000; ++ i) {
            worker->Put(start, 0.01);
            start += 1;
            start %= static_cast<MLTask*>(info.get_task())->get_dimensions();
        }
    });

    // A Single Task
    int kv2 = kvstore::KVStore::Get().CreateKVStore<float>();
    auto task2 = TaskFactory::Get().CreateTask<GenericMLTask>();
    task2.set_dimensions(5);
    task2.set_kvstore(kv2);
    task2.set_running_type(Task::Type::SingleTaskType);  // set the running type explicitly
    engine.AddTask(task2, [](const Info& info) {
        auto& worker = info.get_mlworker();
        // int k = 3;
        // worker->Put(k, 0.456);
        // float v = worker->Get(k);
        // base::log_msg("k: " + std::to_string(k) + " v: " + std::to_string(v));
        int start = info.get_cluster_id();
        for (int i = 0; i < 10000; ++ i) {
            worker->Put(start, 0.01);
            start += 1;
            start %= static_cast<MLTask*>(info.get_task())->get_dimensions();
        }
    });

    // A PS Task
    // int kv3 = kvstore::KVStore::Get().CreateKVStore<float>(kvstore::KVServerDefaultAddHandle<float>());
    int kv3 = kvstore::KVStore::Get().CreateKVStore<float>(kvstore::KVServerBSPHandle<float>(4, true));  // for bsp server
    auto task3 = TaskFactory::Get().CreateTask<GenericMLTask>();
    task3.set_dimensions(5);
    task3.set_kvstore(kv3);
    task3.set_running_type(Task::Type::PSTaskType);  // set the running type explicitly
    task3.set_num_workers(4);  // 4 workers
    engine.AddTask(task3, [kv3](const Info& info) {
        husky::LOG_I << "PS Model running";
        auto& worker = info.get_mlworker();
        for (int i = 0; i < 10; ++ i) {
            std::vector<float> rets;
            std::vector<int> keys{0};
            // pull
            worker->Pull(keys, &rets);
            husky::LOG_I << BLUE("id:"+std::to_string(info.get_local_id())+" iter "+std::to_string(i)+": "+std::to_string(rets[0]));
            // push
            std::vector<float> vals{2.0};
            worker->Push(keys, vals);
        }
    });

    engine.Submit();
    engine.Exit();
    // Stop the kvstore, should stop before mailbox is down
    kvstore::KVStore::Get().Stop();
}
