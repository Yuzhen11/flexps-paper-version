#include "gtest/gtest.h"

#include "kvstore/kvstore.hpp"

namespace husky {
namespace {

class TestBSPServer: public testing::Test {
   public:
    TestBSPServer() {}
    ~TestBSPServer() {}

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

TEST_F(TestBSPServer, Consistency) {
    std::map<std::string, std::string> hint =
    {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kConsistency, husky::constants::kBSP},
        {husky::constants::kNumWorkers, "2"},
    };

    // Start KVStore with 3 servers on each process
    kvstore::KVStore::Get().Start(worker_info, el, zmq_context, 3);

    int kv = kvstore::KVStore::Get().CreateKVStore<int>(hint, 18, 4);
    // num_servers: 3, chunk_size: 2, max_key: 9
    // the result should be:
    // 5 chunks
    // {2, 2, 1}
    // {[0, 4), [4, 8), [8, 9)}
    
    std::thread th1([kv]() {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(0);
        std::vector<husky::constants::Key> keys{0};
        std::vector<int> vals{1};
        std::vector<int> res;
        for (int i = 0; i < 100; ++ i) {
            kvworker->Wait(kv, kvworker->Pull(kv, keys, &res));
            EXPECT_EQ(res[0], 2*i);
            kvworker->Wait(kv, kvworker->Push(kv, keys, vals));
        }
    });
    std::thread th2([kv]() {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(1);
        std::vector<husky::constants::Key> keys{0};
        std::vector<int> vals{1};
        std::vector<int> res;
        for (int i = 0; i < 100; ++ i) {
            kvworker->Wait(kv, kvworker->Pull(kv, keys, &res));
            EXPECT_EQ(res[0], 2*i);
            kvworker->Wait(kv, kvworker->Push(kv, keys, vals));
        }
    });
    th1.join();
    th2.join();
    kvstore::KVStore::Get().Stop();
}

}  // namespace
}  // namespace husky
