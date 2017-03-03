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
        const std::vector<float>& vals, bool is_assign = false) {
    // Pull old_res
    std::vector<float> old_res;
    int ts = kvworker->Pull(kv, keys, &old_res);
    kvworker->Wait(kv, ts);

    // Push
    ts = kvworker->Push(kv, keys, vals);
    kvworker->Wait(kv, ts);

    // Pull new_res
    std::vector<float> new_res;
    ts = kvworker->Pull(kv, keys, &new_res);
    kvworker->Wait(kv, ts);

    EXPECT_EQ(vals.size(), old_res.size());
    EXPECT_EQ(old_res.size(), new_res.size());

    std::vector<float> expect_res(old_res.size());
    if (!is_assign) {
        // if update store by Add method, expect new_res = old_res + vals
        for(size_t i = 0; i < vals.size(); i++) {
            expect_res[i] = old_res[i] + vals[i];
        }
        EXPECT_EQ(new_res, expect_res);
    } else {
        // if update store by assign method, expect new_res = vals
        EXPECT_EQ(new_res, vals);
    }
}

void TestPushPullChunk(int kv,
    kvstore::KVWorker* kvworker,
    const std::vector<size_t>& chunk_ids,
    const std::vector<std::vector<float>*>& push_chunks, bool is_assign = false) {
    // PullChunk
    std::vector<std::vector<float>*> old_res{push_chunks.size()};
    std::vector<std::vector<float>*> new_res{push_chunks.size()};
    int pos = 0;
    for(auto & chunk:push_chunks) {
        old_res[pos] = new std::vector<float>(chunk->size());
        new_res[pos] = new std::vector<float>(chunk->size());
        pos++;
    }

    // Pullchunk old_res
    int ts = kvworker->PullChunks(kv, chunk_ids, old_res);
    kvworker->Wait(kv, ts);

    // PushChunk
    ts = kvworker->PushChunks(kv, chunk_ids, push_chunks);
    kvworker->Wait(kv, ts);

    // PullChunk 
    ts = kvworker->PullChunks(kv, chunk_ids, new_res);
    kvworker->Wait(kv, ts);
    
    EXPECT_EQ(push_chunks.size(), new_res.size());
    EXPECT_EQ(push_chunks.size(), old_res.size());


    for (int i = 0; i < push_chunks.size(); i++) {
        // if update store by Add method, expect new_res = old_res + push_chunks
        if (!is_assign) {
            for(int j = 0; j < (*old_res[i]).size(); j++) {
                (*old_res[i])[j] += (*push_chunks[i])[j];
            }
            EXPECT_EQ(*new_res[i], *old_res[i]);
        } else {
            // if update store by assign method, expect new_res = push_chunks
            EXPECT_EQ(*new_res[i], *push_chunks[i]);
        }
    }

    for(auto& item:old_res) {
        delete item;
    }

    for(auto& item:new_res) {
        delete item;
    }
}

void PushPullUnorderedMap(const WorkerInfo& worker_info, zmq::context_t* zmq_context,
    MailboxEventLoop* el, const std::map<std::string, std::string>& hint = {}, bool is_assign = false) {
    // Start KVStore with 3 servers on each process
    kvstore::KVStore::Get().Start(worker_info, el, zmq_context, 3);

    auto* kvworker1 = kvstore::KVStore::Get().get_kvworker(0);
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>(hint, 18, 4);
    // num_servers: 3, chunk_size: 4, max_key: 18
    // the result should be:
        // 5 chunks
    // {2, 2, 1}
    // {[0, 8), [8, 16), [16, 18)}
    
    // server 1
    TestPushPull(kv1, kvworker1, {1,2},{0.1,0.2}, is_assign);
    // server 2
    TestPushPull(kv1, kvworker1, {9,10},{0.1,0.2}, is_assign);
    // // server 3
    TestPushPull(kv1, kvworker1, {16,17},{0.1,0.2}, is_assign);
    // server 1,2
    TestPushPull(kv1, kvworker1, {3,9},{0.1,0.2}, is_assign);
    // server 1,3
    TestPushPull(kv1, kvworker1, {0,16},{0.1,0.2}, is_assign);
    // server 2,3
    TestPushPull(kv1, kvworker1, {8,16,17},{0.1,0.2,0.3}, is_assign);
    // server 1,2,3
    TestPushPull(kv1, kvworker1, {0,1,2,8,15,17},{0.1,0.2,0.3,0.4,0.5,0.6}, is_assign);

    // Stop KVStore
    kvstore::KVStore::Get().Stop();
}

void PushPullVector(const WorkerInfo& worker_info, zmq::context_t* zmq_context,
    MailboxEventLoop* el, const std::map<std::string, std::string>& hint = {}, bool is_assign = false) {
    // Start KVStore with 3 servers on each process
    kvstore::KVStore::Get().Start(worker_info, el, zmq_context, 3);

    auto* kvworker2 = kvstore::KVStore::Get().get_kvworker(0);
    int kv2 = kvstore::KVStore::Get().CreateKVStore<float>(hint, 18,4);
    // num_servers: 3, chunk_size: 4, max_key: 18
    // the result should be:
    // 5 chunks
    // {2, 2, 1}
    // {[0, 8), [8, 16), [16, 18)}

    // server 1 chunk 0
    std::vector<float> params0{0,1,2,3};
    std::vector<std::vector<float>*> push_chunks0{&params0};
    TestPushPullChunk(kv2, kvworker2, {0}, push_chunks0, is_assign);

    // server 1 chunk 0 1
    std::vector<float> params1{0,0,0.1,1};
    std::vector<float> params2{0.1,0.2,0.3,0.4};
    std::vector<std::vector<float>*> push_chunks1{&params1, &params2};
    TestPushPullChunk(kv2, kvworker2, {0,1}, push_chunks1, is_assign);

    // server 1 2 chunk 1 2
    std::vector<float> params3{0.1,0.4,0.1,0.2};
    std::vector<float> params4{0.1,0.2,0.3,0.4};
    std::vector<std::vector<float>*> push_chunks2{&params3, &params4};
    TestPushPullChunk(kv2, kvworker2, {1,2}, push_chunks2, is_assign);

    // server 2,3 chunk 3 4
    std::vector<float> params5{0.1,0.2,0,0};
    std::vector<float> params6{1,2};
    std::vector<std::vector<float>*> push_chunks3{&params5, &params6};
    TestPushPullChunk(kv2, kvworker2, {3,4}, push_chunks3, is_assign);

    // server 1,2,3 chunk 1 2 4
    std::vector<float> params7{0.1,0.2,0.3,0.4};
    std::vector<float> params8{0,0,0.1,0.2};
    std::vector<float> params9{0.3,0.4};
    std::vector<std::vector<float>*> push_chunks4{&params7,&params8,&params9};
    TestPushPullChunk(kv2, kvworker2, {1,2,4}, push_chunks4, is_assign);

    // Stop KVStore
    kvstore::KVStore::Get().Stop();
}

TEST_F(TestBasicServer, PushPullUnorderedMapAdd) {
    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kUpdateType, husky::constants::kAddUpdate}
    };

    PushPullUnorderedMap(worker_info, zmq_context, el, hint, false);
}

TEST_F(TestBasicServer, PushPullUnorderedMapAssign) {
    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kUpdateType, husky::constants::kAssignUpdate}
    };
    
    PushPullUnorderedMap(worker_info, zmq_context, el, hint, true);
}

// Test Server when change the storage(unordered_map->vector)
TEST_F(TestBasicServer, PushPullVectorAdd) {
    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kStorageType,husky::constants::kVectorStorage},
        {husky::constants::kUpdateType, husky::constants::kAddUpdate}
    };
    
    PushPullUnorderedMap(worker_info, zmq_context, el, hint, false);
}

// Test Server when change the storage(unordered_map->vector)
TEST_F(TestBasicServer, PushPullVectorAssign) {
    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kUpdateType, husky::constants::kAssignUpdate}
    };
    
    PushPullUnorderedMap(worker_info, zmq_context, el, hint, true);
}

TEST_F(TestBasicServer, PushPullChunkUnorderedMapAdd) {
    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kUpdateType, husky::constants::kAddUpdate}
    };

    PushPullVector(worker_info, zmq_context, el, hint, false);
}

TEST_F(TestBasicServer, PushPullChunkUnorderedMapAssign) {
    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kUpdateType, husky::constants::kAssignUpdate}
    };

    PushPullVector(worker_info, zmq_context, el, hint, true);
}

TEST_F(TestBasicServer, PushPullChunkVectorAdd) {
    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kStorageType, husky::constants::kVectorStorage},
        {husky::constants::kUpdateType, husky::constants::kAddUpdate}
    };

    PushPullVector(worker_info, zmq_context, el, hint, false);
}

TEST_F(TestBasicServer, PushPullChunkVectorAssign) {
    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kStorageType, husky::constants::kVectorStorage},
        {husky::constants::kUpdateType, husky::constants::kAssignUpdate}
    };

    PushPullVector(worker_info, zmq_context, el, hint, true);
}

}  // namespace
}  // namespace husky
