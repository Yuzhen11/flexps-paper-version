#include  "gtest/gtest.h"

#include "boost/thread.hpp"
#include "husky/core/mailbox.hpp"
#include "husky/core/worker_info.hpp"
#include "kvstore/kvstore.hpp"
#include "ml/model/chunk_based_mt_model.hpp"

namespace ml {
namespace model {

class TestChunkBasedMTModel : public testing::Test {
   public:
    TestChunkBasedMTModel() {}
    ~TestChunkBasedMTModel() {}

   protected:
    void SetUp() {
       // 1. Create WorkerInfo
       worker_info.add_worker(0, 0, 0);
       worker_info.add_worker(0, 1, 1);
       worker_info.add_worker(0, 2, 2);
       worker_info.set_process_id(0);

       // 2. Create Mailbox
       el = new husky::MailboxEventLoop(&zmq_context);
       el->set_process_id(0);
       recver = new husky::CentralRecver(&zmq_context, "inproc://test");

       // 3. Start and create KVStore
       kvstore::KVStore::Get().Start(worker_info, el, &zmq_context, 1);
       kv = kvstore::KVStore::Get().CreateKVStore<float>();

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

TEST_F(TestChunkBasedMTModel, Start) {}  // For Setup and TearDown

TEST_F(TestChunkBasedMTModel, Prepare) {
    ChunkBasedMTLockModel model(kv, num_params);

    std::vector<husky::constants::Key> some_keys(num_params/2);
    for (int i = 0; i < some_keys.size(); ++i) { some_keys[i] = i; }

    model.Prepare(some_keys, 0);
}

TEST_F(TestChunkBasedMTModel, PullPush) {
    ChunkBasedMTLockModel model(kv, num_params);
    
    std::vector<husky::constants::Key> all_keys(num_params);
    for (int i = 0; i < num_params; ++i) { all_keys[i] = i; }

    std::vector<float> vals(num_params, 1.0);
    std::vector<float> res;

    model.Pull(all_keys, &res, 0);
    model.Push(all_keys, vals);
    model.Pull(all_keys, &res, 0);
    EXPECT_EQ(vals, res);
}

void push_pull_job(ChunkBasedMTLockModel* model, std::vector<husky::constants::Key> keys, int local_id) {
    std::vector<float> res;
    std::vector<float> update(keys.size(), 1.0);
    for (int i = 0; i < 10; ++i) {
        model->Pull(keys, &res, local_id);
        model->Push(keys, update);
    }
}

TEST_F(TestChunkBasedMTModel, MTPushPull) {
    ChunkBasedMTLockModel model(kv, num_params);

    // all keys
    std::vector<husky::constants::Key> all_keys(num_params);
    for (int i = 0; i < num_params; ++i) { all_keys[i] = i; }

    // odd keys
    std::vector<husky::constants::Key> odd_keys(num_params / 2);
    for (int i = 0; i < num_params / 2; ++i) { odd_keys[i] = 2 * i + 1; }

    // even keys
    std::vector<husky::constants::Key> even_keys(num_params / 2);
    for (int i = 0; i < num_params / 2; ++i) { even_keys[i] = 2 * i; }

    boost::thread t1(push_pull_job, &model, all_keys, 0);
    boost::thread t2(push_pull_job, &model, odd_keys, 1);
    boost::thread t3(push_pull_job, &model, even_keys, 2);

    t1.join();
    t2.join();
    t3.join();

    std::vector<float> res;
    model.Pull(all_keys, &res, 0);
    EXPECT_EQ(res, std::vector<float>(all_keys.size(), 20.0));
}
}  // namespace model
}  // namespace ml
