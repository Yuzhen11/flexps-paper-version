#include "gtest/gtest.h"

#include "husky/core/mailbox.hpp"
#include "husky/core/worker_info.hpp"
#include "kvstore/kvstore.hpp"
#include "ml/model/model_with_cm.hpp"

namespace ml {
namespace model {

void push_pull_job(ModelWithCM * model, std::vector<husky::constants::Key> keys, int local_id) {
    std::vector<float> res;
    std::vector<float> update(keys.size(), 1.0);
    for (int i = 0; i < 10; ++i) {
        model->Pull(keys, &res, local_id);
        model->Push(keys, update);
    }
}

class TestChunkFileEditor : public testing::Test {
   public:
    TestChunkFileEditor() {}
    ~TestChunkFileEditor() {}

   protected:
    void SetUp() {}
    void TearDown() {}
};

class TestModelWithCM : public testing::Test {
   public:
    TestModelWithCM() {}
    ~TestModelWithCM() {}

   protected:
    void SetUp() {
        zmq_context = new zmq::context_t;
        // 1. Create WorkerInfo
        worker_info.add_worker(0, 0, 0);  // process id, global id, local id
        worker_info.add_worker(0, 1, 1);
        worker_info.add_worker(0, 2, 2);
        worker_info.set_process_id(0);

        // 2. Create Mailbox
        el = new husky::MailboxEventLoop(zmq_context);
        el->set_process_id(0);
        recver = new husky::CentralRecver(zmq_context, "inproc://test");

        // 3. Start and create KVStore
        kvstore::KVStore::Get().Start(worker_info, el, zmq_context, 1);
        kv = kvstore::KVStore::Get().CreateKVStore<float>();

        // 4. Set RangeManager
        kvstore::RangeManager::Get().SetMaxKeyAndChunkSize(kv, num_params, chunk_size);
    }

    void TearDown() {
        // Stop KVStore
        kvstore::KVStore::Get().Stop();
        delete el;
        delete recver;
        delete zmq_context;
    }

    int num_params = 105;
    int chunk_size = 10;

    int kv = 0;
    husky::WorkerInfo worker_info;
    zmq::context_t* zmq_context;
    husky::MailboxEventLoop* el;
    husky::CentralRecver* recver;
};

TEST_F(TestChunkFileEditor, ReadWrite) {
    int num_chunks = 10;
    int chunk_size = 20;
    int last_chunk_size = 12;

    std::vector<std::vector<float>> chunks(num_chunks); // 20 entries * 9 chunks + 12 entries * 1 chunk
    for (int j = 0; j < num_chunks - 1; ++j) {
        chunks[j].resize(chunk_size);
        for (int i = 0; i < chunk_size; ++i) chunks[j][i] = i + j * chunk_size;
    }
    chunks[num_chunks - 1].resize(last_chunk_size);
    for (int i = 0; i < last_chunk_size; ++i) chunks[num_chunks - 1][i] = i + chunk_size * (num_chunks - 1);

    auto origin = chunks;

    std::vector<size_t> w_ids {2, 4, 5, 8, 9};
    std::vector<size_t> r_ids {2, 5, 9};

    ChunkFileEditor edi(&chunks, chunk_size, last_chunk_size, num_chunks);
    edi.write_chunks(w_ids);
    for (auto id : w_ids) {
        chunks[id].clear();
    }
    edi.read_chunks(r_ids);
    edi.write_chunks(r_ids);
    for (auto id : r_ids) {
        chunks[id].clear();
    }
    edi.read_chunks(w_ids);

    EXPECT_EQ(chunks, origin);
}

TEST_F(TestModelWithCM, Start) {}  // For Setup and TearDown

void test_model_push_pull (ModelWithCM* model) {
    std::vector<husky::constants::Key> keys1{0, 10, 20, 30};
    std::vector<husky::constants::Key> keys2{10, 20, 30, 40, 50};
    std::vector<husky::constants::Key> keys3{30, 40, 50, 60, 70};
    std::vector<float> ans_val{2, 2, 3, 2, 2};

    std::vector<float> res;
    std::vector<float> vals1(keys1.size(), 1.0);
    std::vector<float> vals2(keys2.size(), 1.0);
    std::vector<float> vals3(keys3.size(), 1.0);
    model->Pull(keys1, &res, 0);
    model->Push(keys1, vals1);
    model->Pull(keys2, &res, 0);
    model->Push(keys2, vals2);
    model->Pull(keys3, &res, 0);
    model->Push(keys3, vals3);
    model->Pull(keys2, &res, 0);
    EXPECT_EQ(res, ans_val);
}

TEST_F(TestModelWithCM, PushPullLFU) {
    ModelWithCMLFU model(kv, num_params, 5);  // threshold set to 5
    test_model_push_pull(&model);
}

TEST_F(TestModelWithCM, PushPullLRU) {
    ModelWithCMLRU model(kv, num_params, 5);  // threshold set to 5
    test_model_push_pull(&model);
}

TEST_F(TestModelWithCM, PushPullRandom) {
    ModelWithCMRandom model(kv, num_params, 5);  // threshold set to 5
    test_model_push_pull(&model);
}

}  // namespace model
}  // namespace ml
