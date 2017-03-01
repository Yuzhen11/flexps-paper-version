#include "gtest/gtest.h"

#include "ml/mlworker/single.hpp"

#include "core/instance.hpp"
#include "core/info.hpp"
#include "core/utility.hpp"

namespace husky {
namespace {
/*
 * Test using IntegralModel and ChunkBasedModel
 * TODO: test with ModelTransferManager
 */

class TestSingle: public testing::Test {
   public:
    TestSingle() {}
    ~TestSingle() {}

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
        // 3. Start KVStore
        kvstore::KVStore::Get().Start(worker_info, el, zmq_context);
    }
    void TearDown() {
        kvstore::KVStore::Get().Stop();
        delete el;
        delete recver;
        delete zmq_context;
    }

    WorkerInfo worker_info;
    zmq::context_t* zmq_context;
    MailboxEventLoop* el;
    CentralRecver * recver;
};

TEST_F(TestSingle, Construct) {
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>({}, 9, 2);
    // Create a task
    husky::MLTask task(0);
    task.set_total_epoch(1);
    task.set_dimensions(9);
    task.set_kvstore(kv1);
    // Create an Instance
    husky::Instance instance;
    instance.add_thread(0, 0, 0);  // pid, tid, cid
    instance.set_task(task);
    // Create an Info
    husky::Info info = husky::utility::instance_to_info(instance, worker_info, {0, 0});
    info.set_task(&task);
    // Create SingleWorker
    ml::mlworker::SingleWorker worker(info);
}

void testPushPull(ml::mlworker::SingleWorker& worker) {
    // PushPull
    std::vector<husky::constants::Key> keys = {1,3,5};
    std::vector<float> vals;
    std::vector<float> params = {0,0,0};
    worker.Pull(keys, &vals);
    EXPECT_EQ(vals, params);
    for (int i = 0; i < vals.size(); ++ i) vals[i] = 0.1;
    worker.Push(keys, vals);
    worker.Pull(keys, &vals);
    params = {0.1,0.1,0.1};
    EXPECT_EQ(vals, params);
}
void testV2(ml::mlworker::SingleWorker& worker) {
    // v2 APIs
    std::vector<husky::constants::Key> keys = {1,3,5};
    worker.Prepare_v2(keys);
    EXPECT_EQ(worker.Get_v2(0), float(0));
    EXPECT_EQ(worker.Get_v2(1), float(0));
    worker.Update_v2(1, 0.1);
    EXPECT_EQ(worker.Get_v2(1), float(0.1));
}

TEST_F(TestSingle, Integral) {
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>({}, 9, 2);
    // Create a task
    husky::MLTask task(0);
    task.set_total_epoch(1);
    task.set_dimensions(9);
    task.set_kvstore(kv1);
    // Create an Instance
    husky::Instance instance;
    instance.add_thread(0, 0, 0);  // pid, tid, cid
    instance.set_task(task);
    // Create an Info
    husky::Info info = husky::utility::instance_to_info(instance, worker_info, {0, 0});
    info.set_task(&task);

    // Test Push/Pull API
    {
        ml::mlworker::SingleWorker worker(info);
        testPushPull(worker);
    }
    // Test V2 API
    {
        ml::mlworker::SingleWorker worker(info);
        testV2(worker);
    }
}

TEST_F(TestSingle, Chunk) {
    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kParamType, husky::constants::kChunkType}
    };
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>({}, 9, 2);
    // Create a task
    husky::MLTask task(0);
    task.set_total_epoch(1);
    task.set_dimensions(9);
    task.set_kvstore(kv1);
    task.set_hint(hint);  // set hint
    // Create an Instance
    husky::Instance instance;
    instance.add_thread(0, 0, 0);  // pid, tid, cid
    instance.set_task(task);
    // Create an Info
    husky::Info info = husky::utility::instance_to_info(instance, worker_info, {0, 0});
    info.set_task(&task);

    // Test Push/Pull API
    {
        ml::mlworker::SingleWorker worker(info);
        testPushPull(worker);
    }
    // Test V2 API
    {
        ml::mlworker::SingleWorker worker(info);
        testV2(worker);
    }
}

}  // namespace
}  // namespace husky
