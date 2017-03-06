#include "gtest/gtest.h"

#include "boost/thread.hpp"
#include "husky/core/mailbox.hpp"
#include "husky/core/worker_info.hpp"
#include "kvstore/kvstore.hpp"
#include "ml/model/chunk_based_frequency_model.hpp"

namespace ml {
namespace model {

void push_pull_job(ChunkBasedFrequencyModel* model, std::vector<husky::constants::Key> keys, int local_id) {
    std::vector<float> res;
    std::vector<float> update(keys.size(), 1.0);
    for (int i = 0; i < 10; ++i) {
        model->Pull(keys, &res, local_id);
        model->Push(keys, update);
    }
}

class TestFrequencyModel : public testing::Test {
   public:
    TestFrequencyModel() {}
    ~TestFrequencyModel() {}

   protected:
    void SetUp() {
        // 1. Create WorkerInfo
        worker_info.add_worker(0, 0, 0);  // process id, global id, local id
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
        // Stop KVStore
        kvstore::KVStore::Get().Stop();
        delete el;
        delete recver;
    }

    int num_params = 100;
    int chunk_size = 10;

    int kv = 0;
    husky::WorkerInfo worker_info;
    zmq::context_t zmq_context;
    husky::MailboxEventLoop* el;
    husky::CentralRecver* recver;
};

TEST_F(TestFrequencyModel, Start) {}  // For Setup and TearDown

TEST_F(TestFrequencyModel, LoadFrequentPull) {
    ChunkBasedFrequencyModel model(kv, num_params);

    // frequent keys
    std::vector<husky::constants::Key> frequent_keys(10);
    for (int i = 0; i < 10; ++i) { frequent_keys[i] = 11 * i; }

    std::vector<float> params;

    model.LoadFrequent(0, frequent_keys);
    model.Pull(frequent_keys, &params, 0);
    EXPECT_EQ(std::vector<float>(10, 0), params);
}

TEST_F(TestFrequencyModel, PushPull) {
    ChunkBasedFrequencyModel model(kv, num_params);

    // all keys
    std::vector<husky::constants::Key> all_keys(num_params);
    for (int i = 0; i < num_params; ++i) { all_keys[i] = i; }

    std::vector<float> vals(all_keys.size(), 1.0);
    std::vector<float> res;

    model.Pull(all_keys, &res, 0);
    model.Push(all_keys, vals);
    model.Pull(all_keys, &res, 0);
    EXPECT_EQ(vals, res);
}

TEST_F(TestFrequencyModel, Dump) {
    ChunkBasedFrequencyModel model1(kv, num_params);
    ChunkBasedFrequencyModel model2(kv, num_params);

    // all keys
    std::vector<husky::constants::Key> all_keys(num_params);
    for (int i = 0; i < num_params; ++i) { all_keys[i] = i; }

    std::vector<float> vals(all_keys.size(), 1.0);
    std::vector<float> res;

    model1.Pull(all_keys, &res, 0);
    model1.Push(all_keys, vals);
    model1.Dump(0, "");
    model2.Pull(all_keys, &res, 1);
    EXPECT_EQ(res, vals);
}

TEST_F(TestFrequencyModel, MTPushPull) {
    ChunkBasedMTLockFrequencyModel model(kv, num_params);

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
