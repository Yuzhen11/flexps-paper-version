#include <vector>

#include "ml/common/mlworker.hpp"
#include "worker/engine.hpp"

#include "core/color.hpp"

using namespace husky;

auto test_mlworker_lambda = [](const Info& info) {
    auto& worker = info.get_mlworker();
    int num_iter = 1001;
    for (int i = 0; i < num_iter; ++ i) {
        std::vector<float> rets;
        std::vector<husky::constants::Key> keys{0};
        // pull
        worker->Pull(keys, &rets);
        if (i % 100 == 0)
            husky::LOG_I << BLUE("id:" + std::to_string(info.get_local_id()) + " iter " + std::to_string(i) + ": " +
                                 std::to_string(rets[0]));
        // push
        std::vector<float> vals{1.0};
        worker->Push(keys, vals);
    }
};

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
    std::map<std::string, std::string> hint = 
    { 
        {husky::constants::kType, husky::constants::kHogwild} 
    };
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>(hint, 10, 10);
    auto task1 = TaskFactory::Get().CreateTask<MLTask>();
    task1.set_dimensions(10);
    task1.set_kvstore(kv1);
    task1.set_hint(hint);  // set the running type explicitly
    task1.set_total_epoch(2);                             // 2 epochs
    task1.set_num_workers(4);                             // 4 workers
    engine.AddTask(task1, [](const Info& info) {
        auto& worker = info.get_mlworker();
        // int k = 3;
        // worker->Put(k, 0.456);
        // float v = worker->Get(k);
        // base::log_msg("k: " + std::to_string(k) + " v: " + std::to_string(v));
        husky::constants::Key start = info.get_cluster_id();
        std::vector<float> vals;
        for (int i = 0; i < 10000; ++i) {
            worker->Pull({start}, &vals);
            worker->Push({start}, {0.01});
            start += 1;
            start %= static_cast<MLTask*>(info.get_task())->get_dimensions();
        }
    });

    // A Single Task
    hint = 
    {
        {husky::constants::kType, husky::constants::kSingle}
    };
    int kv2 = kvstore::KVStore::Get().CreateKVStore<float>(hint, 10, 10);
    kvstore::RangeManager::Get().SetMaxKeyAndChunkSize(kv2, 10);
    auto task2 = TaskFactory::Get().CreateTask<MLTask>();
    task2.set_dimensions(10);
    task2.set_kvstore(kv2);
    task2.set_hint(hint);
    engine.AddTask(task2, [](const Info& info) {
        auto& worker = info.get_mlworker();
        worker->Push({2}, {3});
        std::vector<float> res;
        worker->Pull({2}, &res);
        assert(res[0] == 3);
    });

    // A PS Task
    // BSP
    hint = 
    {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kConsistency, husky::constants::kBSP},
        {husky::constants::kNumWorkers, "4"}
    };
    int kv3 = kvstore::KVStore::Get().CreateKVStore<float>(hint, 10, 10);  // for bsp server
    auto task3 = TaskFactory::Get().CreateTask<MLTask>();
    task3.set_dimensions(10);
    task3.set_kvstore(kv3);
    task3.set_hint(hint);
    task3.set_num_workers(4);                           // 4 workers
    engine.AddTask(task3, [](const Info& info) {
        if (info.get_cluster_id() == 0)
            husky::LOG_I << "PS BSP Model running";
        test_mlworker_lambda(info);
    });

    // SSP
    hint = 
    {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kConsistency, husky::constants::kSSP},
        {husky::constants::kNumWorkers, "4"},
        {husky::constants::kStaleness, "1"}
    };
    int kv4 = kvstore::KVStore::Get().CreateKVStore<float>(hint, 10, 10);
    auto task4 = TaskFactory::Get().CreateTask<MLTask>();
    task4.set_dimensions(5);
    task4.set_kvstore(kv4);
    task4.set_hint(hint);
    task4.set_num_workers(4);                           // 4 workers
    engine.AddTask(task4, [](const Info& info) {
        if (info.get_cluster_id() == 0)
            husky::LOG_I << "PS SSP Model running";
        test_mlworker_lambda(info);
    });

    // ASP
    hint = 
    {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kConsistency, husky::constants::kASP},
        {husky::constants::kNumWorkers, "4"}
    };
    int kv5 = kvstore::KVStore::Get().CreateKVStore<float>(hint, 10, 10);
    auto task5 = TaskFactory::Get().CreateTask<MLTask>();
    task5.set_dimensions(5);
    task5.set_kvstore(kv5);
    task5.set_hint(hint);
    task5.set_num_workers(4);                           // 4 workers
    engine.AddTask(task5, [](const Info& info) {
        if (info.get_cluster_id() == 0)
            husky::LOG_I << "PS ASP Model running";
        test_mlworker_lambda(info);
    });

    //  A SPMT Task
    hint = 
    {
        {husky::constants::kType, husky::constants::kSPMT},
        {husky::constants::kConsistency, husky::constants::kASP}
    };
    int kv6 = kvstore::KVStore::Get().CreateKVStore<float>(hint, 10, 10);
    auto task6 = TaskFactory::Get().CreateTask<MLTask>();
    task6.set_dimensions(10);
    task6.set_kvstore(kv6);
    task6.set_hint(hint);
    task6.set_num_workers(4);                             // 4 workers
    engine.AddTask(task6, [](const Info& info) {
        test_mlworker_lambda(info);
    });

    engine.Submit();
    engine.Exit();
    // Stop the kvstore, should stop before mailbox is down
    kvstore::KVStore::Get().Stop();
}
