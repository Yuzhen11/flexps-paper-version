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

        // 3. Start KVStore
        kvstore::KVStore::Get().Start(worker_info, el,
                                      &zmq_context);
    }
    void TearDown() {
        kvstore::KVStore::Get().Stop();
        delete el;
        delete recver;
    }

    WorkerInfo worker_info;
    zmq::context_t zmq_context;
    MailboxEventLoop* el;
    CentralRecver * recver;
};

TEST_F(TestKVStore, Start) {
    // For Setup and TearDown
}

TEST_F(TestKVStore, Create) {
    auto* kvworker = kvstore::KVStore::Get().get_kvworker(0);
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
    EXPECT_EQ(kv1, 0);
}

TEST_F(TestKVStore, PushPull) {
    auto* kvworker = kvstore::KVStore::Get().get_kvworker(0);
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();

    // Push
    std::vector<husky::constants::Key> keys{1,2};
    std::vector<float> vals{3,3};
    int ts = kvworker->Push(kv1, keys, vals);
    kvworker->Wait(kv1, ts);

    // Pull
    std::vector<float> res;
    ts = kvworker->Pull(kv1, keys, &res);
    kvworker->Wait(kv1, ts);
    EXPECT_EQ(res, vals);
}

}  // namespace
}  // namespace husky
