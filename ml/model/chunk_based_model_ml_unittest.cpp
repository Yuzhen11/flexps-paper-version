#include  "gtest/gtest.h"

#include "boost/thread.hpp"
#include "husky/core/mailbox.hpp"
#include "husky/core/worker_info.hpp"
#include "kvstore/kvstore.hpp"
#include "ml/model/chunk_based_model.hpp"

namespace ml {
namespace model {

class TestChunkBasedModel : public testing::Test {
   public:
    TestChunkBasedModel() {}
    ~TestChunkBasedModel() {}

   protected:
    void SetUp() {
       // 1. Create WorkerInfo
       worker_info.add_worker(0, 0, 0);
       worker_info.set_process_id(0);

       // 2. Create Mailbox
       el = new husky::MailboxEventLoop(&zmq_context);
       el->set_process_id(0);
       recver = new husky::CentralRecver(&zmq_context, "inproc://test");

       // 3. Start and create KVStore
       kvstore::KVStore::Get().Start(worker_info, el, &zmq_context, 1);
       kv = kvstore::KVStore::Get().CreateKVStore<float>("default_assign_map", -1, -1);

       // 4. Set RangeManager
       kvstore::RangeManager::Get().SetMaxKeyAndChunkSize(kv, num_params, chunk_size);
    }

    void TearDown() {
        kvstore::KVStore::Get().Stop();
        delete el;
        delete recver;
    }

    int num_params = 1000;
    int chunk_size = 10;

    int kv = 0;
    husky::WorkerInfo worker_info;
    zmq::context_t zmq_context;
    husky::MailboxEventLoop* el;
    husky::CentralRecver* recver;
};

TEST_F(TestChunkBasedModel, Start) {}  // For Setup and TearDown

TEST_F(TestChunkBasedModel, Prepare) {
    ChunkBasedModel<float> model(kv, num_params);

    std::vector<husky::constants::Key> some_keys(num_params/2);
    for (int i = 0; i < some_keys.size(); ++i) { some_keys[i] = i; }

    model.Prepare(some_keys, 0);
}

TEST_F(TestChunkBasedModel, PullPush) {
    ChunkBasedModel<float> model(kv, num_params);
    
    std::vector<husky::constants::Key> all_keys(num_params);
    for (int i = 0; i < num_params; ++i) { all_keys[i] = i; }

    std::vector<float> vals(num_params, 1.0);
    std::vector<float> res;

    model.Pull(all_keys, &res, 0);
    model.Push(all_keys, vals);
    model.Pull(all_keys, &res, 0);
    EXPECT_EQ(vals, res);
}

}  // namespace model
}  // namespace ml
