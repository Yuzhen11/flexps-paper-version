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
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
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
        const std::vector<float>& vals) {
    // Push
    int ts = kvworker->Push(kv, keys, vals);
    kvworker->Wait(kv, ts);

    // Pull
    std::vector<float> res;
    ts = kvworker->Pull(kv, keys, &res);
    kvworker->Wait(kv, ts);
    EXPECT_EQ(res, vals);
}

TEST_F(TestKVStore, PushPull) {
    // Start KVStore
    kvstore::KVStore::Get().Start(worker_info, el, &zmq_context);

    auto* kvworker = kvstore::KVStore::Get().get_kvworker(0);
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
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
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
    kvstore::RangeManager::Get().SetMaxKeyAndChunkSize(kv1, 9, 2);
    // num_servers: 3, chunk_size: 2, max_key: 9
    // the result should be:
    // 5 chunks
    // {2, 2, 1}
    // {[0, 4), [4, 8), [8, 9)}
    
    // server 1
    TestPushPull(kv1, kvworker, {1,2},{0.1,0.2});
    // server 2
    TestPushPull(kv1, kvworker, {4,5},{0.1,0.2});
    // server 3
    TestPushPull(kv1, kvworker, {8},{0.1});
    // server 1,2
    TestPushPull(kv1, kvworker, {0,4},{0.1,0.2});
    // server 1,3
    TestPushPull(kv1, kvworker, {0,8},{0.1,0.2});

    // Stop KVStore
    kvstore::KVStore::Get().Stop();
}

}  // namespace
}  // namespace husky
