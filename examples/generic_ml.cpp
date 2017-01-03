#include <vector>

#include "ml/common/mlworker.hpp"
#include "worker/engine.hpp"

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
    auto task1 = TaskFactory::Get().create_task(Task::Type::GenericMLTaskType);
    static_cast<MLTask*>(task1.get())->set_dimensions(10);
    static_cast<MLTask*>(task1.get())->set_kvstore(kv1);
    static_cast<GenericMLTask*>(task1.get())->set_running_type(Task::Type::HogwildTaskType);  // set the running type explicitly
    task1->set_total_epoch(2);  // 2 epochs
    task1->set_num_workers(4);  // 4 workers
    engine.AddTask(std::move(task1), [](const Info& info) {
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
    auto task2 = TaskFactory::Get().create_task(Task::Type::GenericMLTaskType);
    static_cast<MLTask*>(task2.get())->set_dimensions(5);
    static_cast<MLTask*>(task2.get())->set_kvstore(kv2);
    static_cast<GenericMLTask*>(task2.get())->set_running_type(Task::Type::SingleTaskType);  // set the running type explicitly
    engine.AddTask(std::move(task2), [](const Info& info) {
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
    int kv3 = kvstore::KVStore::Get().CreateKVStore<float>(kvstore::KVServerDefaultAddHandle<float>());
    auto task3 = TaskFactory::Get().create_task(Task::Type::GenericMLTaskType);
    static_cast<MLTask*>(task3.get())->set_dimensions(5);
    static_cast<MLTask*>(task3.get())->set_kvstore(kv3);
    static_cast<GenericMLTask*>(task3.get())->set_running_type(Task::Type::PSTaskType);  // set the running type explicitly
    task3->set_num_workers(4);  // 4 workers
    engine.AddTask(std::move(task3), [](const Info& info) {
        husky::base::log_msg("PS Model running");
        auto& worker = info.get_mlworker();
        for (int i = 0; i < 100; ++ i) {
            std::vector<int> keys{3};
            std::vector<float> vals{0.456};
            worker->Push(keys, vals);
            worker->Sync();
        }
        std::vector<int> keys{3};
        std::vector<float> vals;
        worker->Pull(keys, &vals);
        worker->Sync();
        husky::base::log_msg("PS Generic result is: "+std::to_string(vals[0]));
    });

    engine.Submit();
    engine.Exit();
    // Stop the kvstore, should stop before mailbox is down
    kvstore::KVStore::Get().Stop();
}
