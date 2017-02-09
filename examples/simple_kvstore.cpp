#include "kvstore/kvstore.hpp"
#include "worker/engine.hpp"

#include "core/color.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Start the kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context(), 2);

    auto task = TaskFactory::Get().CreateTask<Task>(1, 1);
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
    engine.AddTask(task, [kv1](const Info& info) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
        std::vector<husky::constants::Key> keys{0};
        std::vector<float> vals{2.0};
        // int ts = kvworker->PushLocal(kv1, info.get_proc_id(), keys, vals);
        int ts = kvworker->Push(kv1, keys, vals);
        kvworker->Wait(kv1, ts);
        husky::LOG_I << "Push Done!";

        std::vector<float> rets;
        // kvworker->Wait(kv1, kvworker->PullLocal(kv1, info.get_proc_id(), keys, &rets));
        kvworker->Wait(kv1, kvworker->Pull(kv1, keys, &rets));
        husky::LOG_I << rets[0];
    });
    engine.Submit();

    // Test KVStore using KVServerBSPHandle: BSP
    task = TaskFactory::Get().CreateTask<Task>(1, 4);
    int kv2 = kvstore::KVStore::Get().CreateKVStore<float>("BSP:4");
    engine.AddTask(task, [kv2](const Info& info) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
        for (int i = 0; i < 10; ++i) {
            std::vector<float> rets;
            std::vector<husky::constants::Key> keys{0};
            // pull
            kvworker->Wait(kv2, kvworker->Pull(kv2, keys, &rets));  // In BSP, expect to see all the update
            husky::LOG_I << BLUE("id:" + std::to_string(info.get_local_id()) + " iter " + std::to_string(i) + ": " +
                                 std::to_string(rets[0]));
            // push
            std::vector<float> vals{1.0};
            kvworker->Wait(kv2, kvworker->Push(kv2, keys, vals));
        }
    });
    engine.Submit();

    // Test KVStore using KVServerSSPHandle: SSP
    task = TaskFactory::Get().CreateTask<Task>(1, 4);
    int kv3 = kvstore::KVStore::Get().CreateKVStore<float>("SSP:4:1");  // staleness: 1
    engine.AddTask(task, [kv3](const Info& info) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
        for (int i = 0; i < 10; ++i) {
            std::vector<float> rets;
            std::vector<husky::constants::Key> keys{0};
            // pull
            kvworker->Wait(kv3, kvworker->Pull(kv3, keys, &rets));  // In SSP, the difference of parameter in each
                                                                    // worker in the same iter should be at most
                                                                    // staleness*num_workers*2-2?
            husky::LOG_I << GREEN("id:" + std::to_string(info.get_local_id()) + " iter " + std::to_string(i) + ": " +
                                  std::to_string(rets[0]));
            // push
            std::vector<float> vals{1.0};
            kvworker->Wait(kv3, kvworker->Push(kv3, keys, vals));
        }
    });
    engine.Submit();

    // Test KVStore using KVServerDefaultAddHandle: ASP
    task = TaskFactory::Get().CreateTask<Task>(1, 4);
    int kv4 = kvstore::KVStore::Get().CreateKVStore<float>("Add");
    engine.AddTask(task, [kv4](const Info& info) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
        for (int i = 0; i < 10; ++i) {
            std::vector<float> rets;
            std::vector<husky::constants::Key> keys{0};
            // pull
            kvworker->Wait(kv4, kvworker->Pull(kv4, keys, &rets));  // In BSP, expect to see all the update
            husky::LOG_I << BLUE("id:" + std::to_string(info.get_local_id()) + " iter " + std::to_string(i) + ": " +
                                 std::to_string(rets[0]));
            // push
            std::vector<float> vals{1.0};
            kvworker->Wait(kv4, kvworker->Push(kv4, keys, vals));
        }
    });
    engine.Submit();

    task = TaskFactory::Get().CreateTask<Task>(1, 1);
    int kv5 = kvstore::KVStore::Get().CreateKVStore<float>();
    kvstore::KVStore::Get().SetMaxKey(kv5, 98);  // 98 keys
    kvstore::RangeManager::Get().SetChunkSize(kv5, 10);  // chunksize is 10
    engine.AddTask(task, [kv5](const Info& info) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
        std::vector<std::vector<float>> params(10, std::vector<float>(10));
        for (int i = 0; i < 10; ++ i) {
            params[0][i] = 123;
            params[3][i] = 234;
            params[6][i] = 789;
        }
        params[9].resize(8);  // the last one has only 8 keys
        for (int i = 0; i < 8; ++ i) params[9][i] = 1000;
        std::vector<size_t> chunk_ids{0,3,6,9};
        std::vector<std::vector<float>*> chunks{&params[0], &params[3], &params[6], &params[9]};
        int ts = kvworker->PushChunks(kv5, chunk_ids, chunks);
        kvworker->Wait(kv5, ts);

        std::vector<float> v[4];
        v[0].resize(10);
        v[1].resize(10);
        v[2].resize(10);
        v[3].resize(8);
        std::vector<std::vector<float>*> pull_chunks{&v[0], &v[1], &v[2], &v[3]};
        ts = kvworker->PullChunks(kv5, chunk_ids, pull_chunks);
        kvworker->Wait(kv5, ts);
        assert(v[0].size() == 10);
        assert(v[1].size() == 10);
        assert(v[2].size() == 10);
        for (int j = 0; j < 10; ++ j) {
            // husky::LOG_I << GREEN("result: "+std::to_string(v[i][j]));
            assert(v[0][j] == 123);
            assert(v[1][j] == 234);
            assert(v[2][j] == 789);
        }
        assert(v[3].size() == 8);
        for (int j = 0; j < v[3].size(); ++ j) {
            // husky::LOG_I << GREEN("result: "+std::to_string(v[3][j]));
            assert(v[3][j] == 1000);
        }
        husky::LOG_I << GREEN("chunk based Push/Pull checked done");
    });

    engine.Submit();
    engine.Exit();
    // Stop the kvstore, should stop before mailbox is down
    kvstore::KVStore::Get().Stop();
}
