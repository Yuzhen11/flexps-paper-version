#include "gtest/gtest.h"

#include "kvstore/kvstore.hpp"

namespace husky {
namespace {

class TestSSPServer: public testing::Test {
   public:
    TestSSPServer() {}
    ~TestSSPServer() {}

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


TEST_F(TestSSPServer, ConsistencyControlOff) {
    std::map<std::string, std::string> hint =
    {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kConsistency, husky::constants::kSSP},
        {husky::constants::kNumWorkers, "2"},
        {husky::constants::kStaleness, "1"}
    };

    // Start KVStore with 3 servers on each process
    kvstore::KVStore::Get().Start(worker_info, el, zmq_context, 1);

    int kv = kvstore::KVStore::Get().CreateKVStore<float>(hint, 18, 4);
    // num_servers: 3, chunk_size: 2, max_key: 9
    // the result should be:
    // 5 chunks
    // {2, 2, 1}
    // {[0, 4), [4, 8), [8, 9)}
    
    auto* kvworker = kvstore::KVStore::Get().get_kvworker(0);
    // Only one thread doing Push/Pull but will not block since we set consistency_control off
    {
        std::vector<size_t> chunk_ids{0,1};
        std::vector<std::vector<float>> params(2);
        std::vector<std::vector<float>*> vals{&params[0], &params[1]};
        for (int i = 0; i < 10; ++ i) {
            kvworker->Wait(kv, kvworker->PullChunks(kv, chunk_ids, vals, true, true, false));  // the last false means consistency_control off
            kvworker->Wait(kv, kvworker->PushChunks(kv, chunk_ids, vals, true, true, false));
        }
    }
    {
        std::vector<husky::constants::Key> keys{0,1,2};
        std::vector<float> vals;
        for (int i = 0; i < 10; ++ i) {
            kvworker->Wait(kv, kvworker->Pull(kv, keys, &vals, true, true, false));  // the last false means consistency_control off
            kvworker->Wait(kv, kvworker->Push(kv, keys, vals, true, true, false));
        }
    }
    kvstore::KVStore::Get().Stop();
}

TEST_F(TestSSPServer, PullChunksWithMinClock) {
    std::map<std::string, std::string> hint =
    {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kConsistency, husky::constants::kSSP},
        {husky::constants::kNumWorkers, "2"},
        {husky::constants::kStaleness, "1"}
    };

    // Start KVStore with 3 servers on each process
    kvstore::KVStore::Get().Start(worker_info, el, zmq_context, 1);

    int kv = kvstore::KVStore::Get().CreateKVStore<float>(hint, 18, 4);
    // num_servers: 3, chunk_size: 2, max_key: 9
    // the result should be:
    // 5 chunks
    // {2, 2, 1}
    // {[0, 4), [4, 8), [8, 9)}
    
    std::thread th1([kv]() {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(0);
        std::vector<size_t> chunk_ids{0,1};
        std::vector<std::vector<float>> params(2);
        std::vector<std::vector<float>*> vals{&params[0], &params[1]};
        for (int i = 0; i < 10; ++ i) {
            int min_clock;
            kvworker->Wait(kv, kvworker->PullChunksWithMinClock(kv, chunk_ids, vals, &min_clock));
            kvworker->Wait(kv, kvworker->PushChunks(kv, chunk_ids, vals));
            husky::LOG_I << "min_clock: " << min_clock;
        }
    });
    std::thread th2([kv]() {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(1);
        std::vector<size_t> chunk_ids{0,1};
        std::vector<std::vector<float>> params(2);
        std::vector<std::vector<float>*> vals{&params[0], &params[1]};
        for (int i = 0; i < 10; ++ i) {
            int min_clock;
            kvworker->Wait(kv, kvworker->PullChunksWithMinClock(kv, chunk_ids, vals, &min_clock));
            kvworker->Wait(kv, kvworker->PushChunks(kv, chunk_ids, vals));
        }
    });
    th1.join();
    th2.join();
    kvstore::KVStore::Get().Stop();
}

}  // namespace
}  // namespace husky
