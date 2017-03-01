#include "gtest/gtest.h"

#include "kvstore/kvstore.hpp"

namespace husky {
namespace {

class TestBasicServer: public testing::Test {
   public:
    TestBasicServer() {}
    ~TestBasicServer() {}

   protected:
    void SetUp() {
        zmq_context = new zmq::context_t;
        // 1. Create WorkerInfo
        worker_info.add_worker(0,0,0);
        worker_info.add_worker(0,1,1);
        worker_info.set_process_id(0);

        // 2. Create Mailbox
        el = new MailboxEventLoop(zmq_context);
        el->set_process_id(0);
        recver = new CentralRecver(zmq_context, "inproc://test");
    }
    void TearDown() {
        delete el;
        delete recver;
        delete zmq_context;
    }

    WorkerInfo worker_info;
    zmq::context_t* zmq_context;
    MailboxEventLoop* el;
    CentralRecver * recver;
};

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

void TestPushPullChunk(int kv,
    kvstore::KVWorker* kvworker,
    const std::vector<size_t>& chunk_ids,
    const std::vector<std::vector<float>*>& push_chunks) {
    // PushChunk
    int ts = kvworker->PushChunks(kv, chunk_ids, push_chunks);
    kvworker->Wait(kv, ts);

    // PullChunk
    std::vector<std::vector<float>*> res{push_chunks.size()};
    int i = 0;
    for (auto& chunk:push_chunks) {
        res[i++] = new std::vector<float>(chunk->size());
    }
    ts = kvworker->PullChunks(kv, chunk_ids, res);
    kvworker->Wait(kv, ts);
    
    EXPECT_EQ(push_chunks.size(), res.size());

    for (int i = 0; i < push_chunks.size(); i++) {
        EXPECT_EQ(*push_chunks[i], *res[i]);
    }

    for(auto& item:res) {
        delete item;
    }
}

// Test Server when change the storage(unordered_map->vector)
TEST_F(TestBasicServer, PushPullUnorderedMap) {
    // Start KVStore with 3 servers on each process
    kvstore::KVStore::Get().Start(worker_info, el, zmq_context, 3);

    auto* kvworker1 = kvstore::KVStore::Get().get_kvworker(0);
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>({}, 18, 4);
    // num_servers: 3, chunk_size: 4, max_key: 18
    // the result should be:
    // 5 chunks
    // {2, 2, 1}
    // {[0, 8), [8, 16), [16, 18)}
    
    // server 1
    TestPushPull(kv1, kvworker1, {1,2},{0.1,0.2});
    // server 2
    TestPushPull(kv1, kvworker1, {9,10},{0.1,0.2});
    // server 3
    TestPushPull(kv1, kvworker1, {16,17},{0.1,0.2});
    // server 1,2
    TestPushPull(kv1, kvworker1, {3,9},{0.1,0.2});
    // server 1,3
    TestPushPull(kv1, kvworker1, {0,16},{0.1,0.2});
    // server 2,3
    TestPushPull(kv1, kvworker1, {8,16,17},{0.1,0.2,0.3});
    // server 1,2,3
    TestPushPull(kv1, kvworker1, {0,1,2,8,15,17},{0.1,0.2,0.3,0.4,0.5,0.6});

    // Stop KVStore
    kvstore::KVStore::Get().Stop();
}

// Test Server when change the storage(unordered_map->vector)
TEST_F(TestBasicServer, PushPullVector) {
    // Start KVStore with 3 servers on each process
    kvstore::KVStore::Get().Start(worker_info, el, zmq_context, 3);

    auto* kvworker1 = kvstore::KVStore::Get().get_kvworker(0);
    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kStorageType, husky::constants::kVectorStorage}
    };
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>(hint, 18, 4);
    // num_servers: 3, chunk_size: 4, max_key: 18
    // the result should be:
    // 5 chunks
    // {2, 2, 1}
    // {[0, 8), [8, 16), [16, 18)}
    
    // server 1
    TestPushPull(kv1, kvworker1, {1,2},{0.1,0.2});
    // server 2
    TestPushPull(kv1, kvworker1, {9,10},{0.1,0.2});
    // server 3
    TestPushPull(kv1, kvworker1, {16,17},{0.1,0.2});
    // server 1,2
    TestPushPull(kv1, kvworker1, {3,9},{0.1,0.2});
    // server 1,3
    TestPushPull(kv1, kvworker1, {0,16},{0.1,0.2});
    // server 2,3
    TestPushPull(kv1, kvworker1, {8,16,17},{0.1,0.2,0.3});
    // server 1,2,3
    TestPushPull(kv1, kvworker1, {0,1,2,8,15,17},{0.1,0.2,0.3,0.4,0.5,0.6});

    // Stop KVStore
    kvstore::KVStore::Get().Stop();
}

TEST_F(TestBasicServer, PushPullChunkUnorderedMap) {
    // Start KVStore with 3 servers on each process
    kvstore::KVStore::Get().Start(worker_info, el, zmq_context, 3);

    auto* kvworker2 = kvstore::KVStore::Get().get_kvworker(0);
    int kv2 = kvstore::KVStore::Get().CreateKVStore<float>({}, 18,4);
    // num_servers: 3, chunk_size: 4, max_key: 18
    // the result should be:
    // 5 chunks
    // {2, 2, 1}
    // {[0, 8), [8, 16), [16, 18)}

    // server 1 chunk 0
    std::vector<float> params0{0,1,2,3};
    std::vector<std::vector<float>*> push_chunks0{&params0};
    TestPushPullChunk(kv2, kvworker2, {0},push_chunks0);

    // server 1 chunk 0 1
    std::vector<float> params1{0,0,0.1,1};
    std::vector<float> params2{0.1,0.2,0.3,0.4};
    std::vector<std::vector<float>*> push_chunks1{&params1, &params2};
    TestPushPullChunk(kv2, kvworker2, {0,1},push_chunks1);

    // server 1 2 chunk 1 2
    std::vector<float> params3{0.1,0.4,0.1,0.2};
    std::vector<float> params4{0.1,0.2,0.3,0.4};
    std::vector<std::vector<float>*> push_chunks2{&params3, &params4};
    TestPushPullChunk(kv2, kvworker2, {1,2},push_chunks2);

    // server 2,3 chunk 3 4
    std::vector<float> params5{0.1,0.2,0,0};
    std::vector<float> params6{1,2};
    std::vector<std::vector<float>*> push_chunks3{&params5, &params6};
    TestPushPullChunk(kv2, kvworker2, {3,4},push_chunks3);

    // server 1,2,3 chunk 1 2 4
    std::vector<float> params7{0.1,0.2,0.3,0.4};
    std::vector<float> params8{0,0,0.1,0.2};
    std::vector<float> params9{0.3,0.4};
    std::vector<std::vector<float>*> push_chunks4{&params7,&params8,&params9};
    TestPushPullChunk(kv2, kvworker2, {1,2,4},push_chunks4);

    // Stop KVStore
    kvstore::KVStore::Get().Stop();
}

TEST_F(TestBasicServer, PushPullChunkVector) {
    // Start KVStore with 3 servers on each process
    kvstore::KVStore::Get().Start(worker_info, el, zmq_context, 3);

    auto* kvworker2 = kvstore::KVStore::Get().get_kvworker(0);
    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kStorageType, husky::constants::kVectorStorage}
    };
    int kv2 = kvstore::KVStore::Get().CreateKVStore<float>(hint, 18,4);
    // num_servers: 3, chunk_size: 4, max_key: 18
    // the result should be:
    // 5 chunks
    // {2, 2, 1}
    // {[0, 8), [8, 16), [16, 18)}

    // server 1 chunk 0
    std::vector<float> params0{0,1,2,3};
    std::vector<std::vector<float>*> push_chunks0{&params0};
    TestPushPullChunk(kv2, kvworker2, {0},push_chunks0);

    // server 1 chunk 0 1
    std::vector<float> params1{0,0,0.1,1};
    std::vector<float> params2{0.1,0.2,0.3,0.4};
    std::vector<std::vector<float>*> push_chunks1{&params1, &params2};
    TestPushPullChunk(kv2, kvworker2, {0,1},push_chunks1);

    // server 1 2 chunk 1 2
    std::vector<float> params3{0.1,0.4,0.1,0.2};
    std::vector<float> params4{0.1,0.2,0.3,0.4};
    std::vector<std::vector<float>*> push_chunks2{&params3, &params4};
    TestPushPullChunk(kv2, kvworker2, {1,2},push_chunks2);

    // server 2,3 chunk 3 4
    std::vector<float> params5{0.1,0.2,0,0};
    std::vector<float> params6{1,2};
    std::vector<std::vector<float>*> push_chunks3{&params5, &params6};
    TestPushPullChunk(kv2, kvworker2, {3,4},push_chunks3);

    // server 1,2,3 chunk 1 2 4
    std::vector<float> params7{0.1,0.2,0.3,0.4};
    std::vector<float> params8{0,0,0.1,0.2};
    std::vector<float> params9{0.3,0.4};
    std::vector<std::vector<float>*> push_chunks4{&params7,&params8,&params9};
    TestPushPullChunk(kv2, kvworker2, {1,2,4},push_chunks4);

    // Stop KVStore
    kvstore::KVStore::Get().Stop();
}

}  // namespace
}  // namespace husky
