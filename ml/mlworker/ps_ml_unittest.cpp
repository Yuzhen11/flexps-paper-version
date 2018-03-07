#include "gtest/gtest.h"

#include "ml/mlworker/ps.hpp"

#include "boost/thread.hpp"
#include "core/instance.hpp"
#include "core/info.hpp"
#include "core/utility.hpp"

namespace husky {
namespace {

/*
 * Test different features:
 * PSChunkChunkWorker: process_cache, chunk-based
 * PSMapChunkWorker:   process_cache, unordered map
 * PSChunkNoneWorker:  chunk-based
 * PSMapNoneWorker:    unordered map
 */
class TestPS: public testing::Test {
   public:
    TestPS() {}
    ~TestPS() {}

   protected:
    void SetUp() {
        zmq_context = new zmq::context_t;
        worker_info = new WorkerInfo;
        // 1. Create WorkerInfo
        worker_info->add_worker(0,0,0);
        worker_info->add_worker(0,1,1);
        worker_info->set_process_id(0);

        // 2. Create Mailbox
        el = new MailboxEventLoop(zmq_context);
        el->set_process_id(0);
        recver = new CentralRecver(zmq_context, "inproc://test");
        // 3. Start KVStore
        kvstore::KVStore::Get().Start(*worker_info, el, zmq_context);
    }

    void TearDown() {
        kvstore::KVStore::Get().Stop();
        delete worker_info;
        delete el;
        delete recver;
        delete zmq_context;
    }

   public:
    int num_params = 100;
    WorkerInfo* worker_info;
    zmq::context_t* zmq_context;
    MailboxEventLoop* el;
    CentralRecver * recver;
};

TEST_F(TestPS, Construct) {
    int num_workers = 2;
    int staleness = 2;
    int dims = 10;
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>("ssp_add_map", num_workers, staleness, dims, 2);
    // Create a task
    husky::Task task(0);
    task.set_total_epoch(1);
    // Create an Instance
    husky::Instance instance;
    instance.add_thread(0, 0, 0);  // pid, tid, cid
    instance.set_task(task);
    // Create an Info
    husky::Info info = husky::utility::instance_to_info(instance, *worker_info, {0, 0}, true);
    // Create a TableInfo
    TableInfo table_info {
        kv1, dims,
        husky::ModeType::PS, 
        husky::Consistency::SSP, 
        husky::WorkerType::PSWorker, 
        husky::ParamType::None,
        staleness
    };
    // Create PSChunkChunkWorker
    ml::mlworker::PSChunkChunkWorker<float> worker1(info, table_info, *zmq_context);
    /*
    // Create PSMapChunkWorker
    ml::mlworker::PSMapChunkWorker<float> worker2(info, *zmq_context);
    */
    // Create PSChunkNoneWorker
    ml::mlworker::PSChunkNoneWorker<float> worker3(info, table_info);
    // Create PSMapNoneWorker
    ml::mlworker::PSMapNoneWorker<float> worker4(info, table_info);
}

void testPushPull(ml::mlworker::GenericMLWorker<float>* worker) {
    // PushPull
    std::vector<husky::constants::Key> keys = {0,10,20,30,40,50,60,70,80,90};
    std::vector<float> old_vals;
    std::vector<float> vals(keys.size(), 0.1);
    worker->Pull(keys, &old_vals);
    worker->Push(keys, vals);
}

void testV2(ml::mlworker::GenericMLWorker<float>* worker) {
    // v2 APIs
    std::vector<husky::constants::Key> keys = {0,10,20,30,40,50,60,70,80,90};
    worker->Prepare_v2(keys);
    std::vector<float> old_vals;
    std::vector<float> vals(keys.size(), 0.1);
    for (int i = 0; i < keys.size(); ++ i)
        old_vals.push_back(worker->Get_v2(i));
    for (int i = 0; i < keys.size(); ++ i)
        worker->Update_v2(i, vals[i]);
    worker->Clock_v2();
}

void test_multiple_threads(TestPS* obj, int type) {
    int num_workers = 2;
    int staleness = 2;
    int dims = 100;
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>("ssp_add_map", num_workers, staleness, dims, 2);
    // Create a task
    husky::Task task(0);
    task.set_total_epoch(2);
    // Create an Instance
    husky::Instance instance;
    instance.add_thread(0, 0, 0);  // pid, tid, cid
    instance.add_thread(0, 1, 1);  // pid, tid, cid
    instance.set_task(task);

    // Create a TableInfo
    TableInfo table_info {
        kv1, dims,
        husky::ModeType::PS, 
        husky::Consistency::SSP, 
        husky::WorkerType::PSWorker, 
        husky::ParamType::None,
        staleness
    };
    int iters = 10;

    boost::thread t1([&instance, &obj, &iters, &type, table_info](){
        husky::Info info = husky::utility::instance_to_info(instance, *obj->worker_info, {0, 0}, true);
        if (type == 3) {
            ml::mlworker::PSChunkChunkWorker<float> worker(info, table_info, *obj->zmq_context);
            for (int i = 0; i < iters; ++i) {
                testPushPull(&worker);
                testV2(&worker);
            }
        } else if (type == 2) {
            ml::mlworker::PSMapChunkWorker<float> worker(info, table_info, *obj->zmq_context);
            for (int i = 0; i < iters; ++i) {
                testPushPull(&worker);
                testV2(&worker);
            }
        } else if (type == 1) {
            ml::mlworker::PSChunkNoneWorker<float> worker(info, table_info);
            for (int i = 0; i < iters; ++i) {
                testPushPull(&worker);
                testV2(&worker);
            }
        } else if (type == 0) {
            ml::mlworker::PSMapNoneWorker<float> worker(info, table_info);
            for (int i = 0; i < iters; ++i) {
                testPushPull(&worker);
                testV2(&worker);
            }
        }
    });
    boost::thread t2([&instance, &obj, &iters, &type, table_info](){
        husky::Info info = husky::utility::instance_to_info(instance, *obj->worker_info, {1, 1}, false);
        if (type == 3) {
            ml::mlworker::PSChunkChunkWorker<float> worker(info, table_info, *obj->zmq_context);
            for (int i = 0; i < iters; ++i) {
                if (i % 3 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
                testPushPull(&worker);
                testV2(&worker);
            }
        } else if (type == 2) {
            ml::mlworker::PSMapChunkWorker<float> worker(info, table_info, *obj->zmq_context);
            for (int i = 0; i < iters; ++i) {
                if (i % 3 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
                testPushPull(&worker);
                testV2(&worker);
            }
        } else if (type == 1) {
            ml::mlworker::PSChunkNoneWorker<float> worker(info, table_info);
            for (int i = 0; i < iters; ++i) {
                if (i % 3 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
                testPushPull(&worker);
                testV2(&worker);
            }
        } else if (type == 0) {
            ml::mlworker::PSMapNoneWorker<float> worker(info, table_info);
            for (int i = 0; i < iters; ++i) {
                if (i % 3 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
                testPushPull(&worker);
                testV2(&worker);
            }
        }
    });
    t1.join();
    t2.join();
}

TEST_F(TestPS, PSChunkChunkWorker) {
    test_multiple_threads(static_cast<TestPS*>(this), 3);
}

TEST_F(TestPS, PSMapChunkWorker) {
    test_multiple_threads(static_cast<TestPS*>(this), 2);
}

TEST_F(TestPS, PSChunkNoneWorker) {
    test_multiple_threads(static_cast<TestPS*>(this), 1);
}

TEST_F(TestPS, PSMapNoneWorker) {
    test_multiple_threads(static_cast<TestPS*>(this), 0);
}

}  // namespace
}  // namespace husky
