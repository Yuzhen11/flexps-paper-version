#include "gtest/gtest.h"

#include "kvstore/kvstore.hpp"

namespace husky {
namespace {

class TestKVStore: public testing::Test {
   public:
    TestKVStore() {}
    ~TestKVStore() {}

   protected:
    void SetUp() {
        // 1. Create WorkerInfo
        worker_info.add_worker(0,0,0);
        worker_info.add_worker(0,1,1);
        worker_info.set_process_id(0);

        // 2. Create Mailbox
        el = new MailboxEventLoop(&zmq_context);
        el->set_process_id(0);
        recver = new CentralRecver(&zmq_context, "inproc://test");
    }
    void TearDown() {
        delete el;
        delete recver;
    }

    WorkerInfo worker_info;
    zmq::context_t zmq_context;
    MailboxEventLoop* el;
    CentralRecver * recver;
};

TEST_F(TestKVStore, StartStop) {
    // Start KVStore
    kvstore::KVStore::Get().Start(worker_info, el, &zmq_context);
    // Stop KVStore
    kvstore::KVStore::Get().Stop();
}

TEST_F(TestKVStore, Create) {
    // Start KVStore
    kvstore::KVStore::Get().Start(worker_info, el, &zmq_context);

    auto* kvworker = kvstore::KVStore::Get().get_kvworker(0);
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>("default_assign_map", -1, -1);
    EXPECT_EQ(kv1, 0);
    EXPECT_EQ(kvstore::RangeManager::Get().GetNumServers(), 1);  // 1 server by default
    EXPECT_EQ(kvstore::RangeManager::Get().GetNumRanges(), 1);  // 1 range

    // Stop KVStore
    kvstore::KVStore::Get().Stop();  // Stop will call RangeManager::Clear()
    EXPECT_EQ(kvstore::RangeManager::Get().GetNumServers(), -1);
    EXPECT_EQ(kvstore::RangeManager::Get().GetNumRanges(), 0);
}

void TestPushPull(int kv,
        kvstore::KVWorker* kvworker,
        const std::vector<husky::constants::Key>& keys, 
        const std::vector<float>& vals, bool send_all = true, bool local_zero_copy = true) {
    // Push
    int ts = kvworker->Push(kv, keys, vals, send_all, local_zero_copy);
    kvworker->Wait(kv, ts);

    // Pull
    std::vector<float> res;
    ts = kvworker->Pull(kv, keys, &res, send_all, local_zero_copy);
    kvworker->Wait(kv, ts);
    EXPECT_EQ(res, vals);
}

TEST_F(TestKVStore, PushPull) {
    // Start KVStore
    kvstore::KVStore::Get().Start(worker_info, el, &zmq_context);

    auto* kvworker = kvstore::KVStore::Get().get_kvworker(0);
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>("default_assign_map", -1, -1);
    // Withcout calling RangeManager::SetMaxKeyAndChunkSize(...), 
    // the max_key is the max, the chunk size is 100

    TestPushPull(kv1, kvworker, {1,2},{3,3});
    TestPushPull(kv1, kvworker, {100,200},{3,3});

    // Stop KVStore
    kvstore::KVStore::Get().Stop();
}

TEST_F(TestKVStore, PushPull2) {
    // Start KVStore with 3 servers on each process
    kvstore::KVStore::Get().Start(worker_info, el, &zmq_context, 3);

    auto* kvworker = kvstore::KVStore::Get().get_kvworker(0);
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>("default_assign_map", -1, -1, 9, 2);
    int kv2 = kvstore::KVStore::Get().CreateKVStore<float>("default_assign_vector", -1, -1, 9, 2);
    // num_servers: 3, chunk_size: 2, max_key: 9
    // the result should be:
    // 5 chunks
    // {2, 2, 1}
    // {[0, 4), [4, 8), [8, 9)}
    
    for (auto kv : {kv1, kv2}) {
        TestPushPull(kv, kvworker, {1,2},{0.1,0.2});
        TestPushPull(kv, kvworker, {4,5},{0.1,0.2});
        TestPushPull(kv, kvworker, {8},{0.1});
        TestPushPull(kv, kvworker, {0,4},{0.1,0.2});
        TestPushPull(kv, kvworker, {0,8},{0.1,0.2});

        for (auto send_all : {true, false}) {
            for (auto local_zero_copy : {true, false}) {
                TestPushPull(kv, kvworker, {1,2},{0.1,0.2}, send_all, local_zero_copy);
                TestPushPull(kv, kvworker, {4,5},{0.1,0.2}, send_all, local_zero_copy);
                TestPushPull(kv, kvworker, {8},{0.1}, send_all, local_zero_copy);
                TestPushPull(kv, kvworker, {0,4},{0.1,0.2}, send_all, local_zero_copy);
                TestPushPull(kv, kvworker, {0,8},{0.1,0.2}, send_all, local_zero_copy);
            }
        }
    }

    // Stop KVStore
    kvstore::KVStore::Get().Stop();
}

void TestPushPullChunks(int kv,
        kvstore::KVWorker* kvworker,
        const std::vector<size_t>& chunk_ids, 
        size_t chunk_size, float val,
        bool send_all = true, bool local_zero_copy = true) {
    std::vector<std::vector<float>> params(chunk_ids.size(), std::vector<float>(chunk_size));
    for (auto& param : params)  {
        for (size_t i = 0; i < chunk_size; ++ i) {
            param[i] = val;
        }
    }
    std::vector<std::vector<float>*> vals(chunk_ids.size());
    for (int i = 0; i < vals.size(); ++ i) {
        vals[i] = &params[i];
    }
    int ts = kvworker->PushChunks(kv, chunk_ids, vals);
    kvworker->Wait(kv, ts);

    std::vector<std::vector<float>> res(chunk_ids.size());
    for (int i = 0; i < vals.size(); ++ i) {
        vals[i] = &res[i];
    }
    ts = kvworker->PullChunks(kv, chunk_ids, vals);
    kvworker->Wait(kv, ts);

    for (int i = 0; i < res.size(); ++ i) {
        EXPECT_EQ(res[i], params[i]);
    }
}

TEST_F(TestKVStore, PushPullChunks) {
    // Start KVStore with 3 servers on each process
    kvstore::KVStore::Get().Start(worker_info, el, &zmq_context, 3);

    auto* kvworker = kvstore::KVStore::Get().get_kvworker(0);
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>("default_assign_map", -1, -1, 10, 2);
    int kv2 = kvstore::KVStore::Get().CreateKVStore<float>("default_assign_vector", -1, -1, 10, 2);
    // num_servers: 3, chunk_size: 2, max_key: 10
    // the result should be:
    // 5 chunks
    // {2, 2, 1}
    // {[0, 4), [4, 8), [8, 10)}
    

    for (auto kv : {kv1, kv2}) {
        TestPushPullChunks(kv, kvworker, {0,1}, 2, 0.1);
        TestPushPullChunks(kv, kvworker, {2,3}, 2, 0.2);
        TestPushPullChunks(kv, kvworker, {4}, 2, 0.3);
        TestPushPullChunks(kv, kvworker, {1,2}, 2, 0.4);

        for (auto send_all : {true, false}) {
            for (auto local_zero_copy : {true, false}) {
                TestPushPullChunks(kv, kvworker, {0,1}, 2, 0.1, send_all, local_zero_copy);
                TestPushPullChunks(kv, kvworker, {2,3}, 2, 0.2, send_all, local_zero_copy);
                TestPushPullChunks(kv, kvworker, {4}, 2, 0.3, send_all, local_zero_copy);
                TestPushPullChunks(kv, kvworker, {1,2}, 2, 0.4, send_all, local_zero_copy);
            }
        }
    }

    // Stop KVStore
    kvstore::KVStore::Get().Stop();
}

}  // namespace
}  // namespace husky
