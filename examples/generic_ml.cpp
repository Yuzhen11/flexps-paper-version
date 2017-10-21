#include <vector>

#include "worker/engine.hpp"
#include "ml/ml.hpp"

#include "core/color.hpp"

using namespace husky;

auto test_mlworker_lambda = [](const Info& info, const TableInfo& table_info) {
    auto worker = ml::CreateMLWorker<float>(info, table_info);
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
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    //  A Hogwild! Task
    int dims1 = 10;
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>("default_assign_map", -1, -1, dims1, 10);
    auto task1 = TaskFactory::Get().CreateTask<MLTask>();
    task1.set_local();
    task1.set_total_epoch(2);                             // 2 epochs
    task1.set_num_workers(4);                             // 4 workers
    TableInfo table_info1 {
        kv1, dims1, 
        husky::ModeType::Hogwild, 
        husky::Consistency::None, 
        husky::WorkerType::None, 
        husky::ParamType::IntegralType
    };
    engine.AddTask(task1, [table_info1](const Info& info) {
        husky::LOG_I << "table info: " << table_info1.DebugString();
        auto worker = ml::CreateMLWorker<float>(info, table_info1);
        husky::constants::Key start = info.get_cluster_id();
        std::vector<float> vals;
        for (int i = 0; i < 10000; ++i) {
            worker->Pull({start}, &vals);
            worker->Push({start}, {0.01});
            start += 1;
            start %= table_info1.dims;
        }
    });

    // A Single Task
    int dims2 = 10;
    int kv2 = kvstore::KVStore::Get().CreateKVStore<float>("default_assign_map", -1, -1, dims2, 10);
    kvstore::RangeManager::Get().SetMaxKeyAndChunkSize(kv2, 10);
    auto task2 = TaskFactory::Get().CreateTask<MLTask>();
    task2.set_local();
    task2.set_num_workers(1);
    TableInfo table_info2 {
        kv2, dims2,
        husky::ModeType::Single, 
        husky::Consistency::None, 
        husky::WorkerType::None, 
        husky::ParamType::IntegralType
    };
    engine.AddTask(task2, [table_info2](const Info& info) {
        auto worker = ml::CreateMLWorker<float>(info, table_info2);
        worker->Push({2}, {3});
        std::vector<float> res;
        worker->Pull({2}, &res);
        assert(res[0] == 3);
    });

    // A PS Task
    // BSP
    int num_workers3 = 4;
    int dims3 = 10;
    int kv3 = kvstore::KVStore::Get().CreateKVStore<float>("bsp_add_map", num_workers3, -1, dims3, 10);  // for bsp server
    auto task3 = TaskFactory::Get().CreateTask<MLTask>();
    task3.set_num_workers(num_workers3);
    TableInfo table_info3 {
        kv3, dims3,
        husky::ModeType::PS, 
        husky::Consistency::BSP, 
        husky::WorkerType::PSWorker, 
        husky::ParamType::None
    };
    engine.AddTask(task3, [table_info3](const Info& info) {
        if (info.get_cluster_id() == 0)
            husky::LOG_I << "PS BSP Model running";
        test_mlworker_lambda(info, table_info3);
    });

    // SSP
    int dims4 = 10;
    int staleness4 = 1;
    int num_workers4 = 4;
    int kv4 = kvstore::KVStore::Get().CreateKVStore<float>("ssp_add_map", num_workers4, staleness4, dims4, 10);
    auto task4 = TaskFactory::Get().CreateTask<MLTask>();
    task4.set_num_workers(num_workers4);
    TableInfo table_info4 {
        kv4, dims4,
        husky::ModeType::PS, 
        husky::Consistency::SSP, 
        husky::WorkerType::PSWorker, 
        husky::ParamType::None,
        staleness4
    };
    engine.AddTask(task4, [table_info4](const Info& info) {
        if (info.get_cluster_id() == 0)
            husky::LOG_I << "PS SSP Model running";
        test_mlworker_lambda(info, table_info4);
    });

    // ASP
    int dims5 = 10;
    int num_workers5 = 4;
    int kv5 = kvstore::KVStore::Get().CreateKVStore<float>("default_add_map", -1, -1, dims5, 10);
    auto task5 = TaskFactory::Get().CreateTask<MLTask>();
    task5.set_num_workers(num_workers5);
    TableInfo table_info5 {
        kv5, dims5,
        husky::ModeType::PS, 
        husky::Consistency::ASP, 
        husky::WorkerType::PSWorker, 
        husky::ParamType::None
    };
    engine.AddTask(task5, [table_info5](const Info& info) {
        if (info.get_cluster_id() == 0)
            husky::LOG_I << "PS ASP Model running";
        test_mlworker_lambda(info, table_info5);
    });

    //  A SPMT Task
    int dims6 = 10;
    int num_workers6 = 4;
    int kv6 = kvstore::KVStore::Get().CreateKVStore<float>("default_assign_map", -1, -1, dims6, 10);
    auto task6 = TaskFactory::Get().CreateTask<MLTask>();
    task6.set_local();
    task6.set_num_workers(num_workers6);
    TableInfo table_info6 {
        kv6, dims6,
        husky::ModeType::SPMT, 
        husky::Consistency::ASP, 
        husky::WorkerType::None, 
        husky::ParamType::IntegralType
    };
    engine.AddTask(task6, [table_info6](const Info& info) {
        test_mlworker_lambda(info, table_info6);
    });

    engine.Submit();
    engine.Exit();
    // Stop the kvstore, should stop before mailbox is down
    kvstore::KVStore::Get().Stop();
}
