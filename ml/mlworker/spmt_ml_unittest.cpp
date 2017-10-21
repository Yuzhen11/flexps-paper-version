#include "gtest/gtest.h"

#include "ml/mlworker/spmt.hpp"
#include "ml/mlworker/hogwild.hpp"

#include "core/instance.hpp"
#include "core/info.hpp"
#include "core/utility.hpp"

namespace husky {
namespace {

/*
 * Test using different combination:
 * {IntegralModel, ChunkBasedModel}
 * {BSP, SSP, ASP, hogwild}
 * {v1, v2_api}
 * TODO: test with ModelTransferManager
 */
class TestSPMT: public testing::Test {
   public:
    TestSPMT() {}
    ~TestSPMT() {}

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
    WorkerInfo* worker_info;
    zmq::context_t* zmq_context;
    MailboxEventLoop* el;
    CentralRecver * recver;
};

TEST_F(TestSPMT, Construct) {
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>("default_assign_map", -1, -1, 9, 2);
    // Create a task
    husky::MLTask task(0);
    // Create an Instance
    husky::Instance instance;
    instance.add_thread(0, 0, 0);  // pid, tid, cid
    instance.set_task(task);
    // Create an Info
    husky::Info info = husky::utility::instance_to_info(instance, *worker_info, {0, 0}, true);
    // Create a TableInfo
    TableInfo table_info {
        kv1, 9,
        husky::ModeType::SPMT, 
        husky::Consistency::ASP, 
        husky::WorkerType::None, 
        husky::ParamType::IntegralType
    };
    // Create SPMTWorker
    ml::mlworker::SPMTWorker<float> worker(info, table_info, *zmq_context);
}

void testPushPull(ml::mlworker::SPMTWorker<float>& worker, bool check = false) {
    // PushPull
    std::vector<husky::constants::Key> keys = {1,3,5};
    std::vector<float> old_vals;
    std::vector<float> vals = {0.1, 0.1, 0.1};
    std::vector<float> new_vals;
    worker.Pull(keys, &old_vals);
    worker.Push(keys, vals);
    worker.Pull(keys, &new_vals);
    if (check) {
        for (int i = 0; i < keys.size(); ++ i)
            EXPECT_EQ(abs(vals[i]+old_vals[i]-new_vals[i]) < 0.0001, true);
    }
    worker.Push(keys, {0, 0, 0});
}
void testV2(ml::mlworker::SPMTWorker<float>& worker, bool check = false) {
    // v2 APIs
    std::vector<husky::constants::Key> keys = {1,3,5};
    worker.Prepare_v2(keys);
    std::vector<float> old_vals;
    std::vector<float> vals = {0.1, 0.1, 0.1};
    std::vector<float> new_vals;
    for (int i = 0; i < keys.size(); ++ i)
        old_vals.push_back(worker.Get_v2(i));
    for (int i = 0; i < keys.size(); ++ i)
        worker.Update_v2(i, vals[i]);
    for (int i = 0; i < keys.size(); ++ i)
        new_vals.push_back(worker.Get_v2(i));
    if (check) {
        for (int i = 0; i < keys.size(); ++ i)
            EXPECT_EQ(abs(vals[i]+old_vals[i]-new_vals[i]) < 0.0001, true);
    }
    worker.Clock_v2();
}

void test_single_thread(const husky::ParamType& param_type, const husky::Consistency& consistency, 
        TestSPMT* obj) {
    bool is_hogwild = false;
    if (consistency == husky::Consistency::None)
        is_hogwild = true;
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>("default_assign_map", -1, -1, 9, 2);
    // Create a task
    husky::MLTask task(0);
    // Create an Instance
    husky::Instance instance;
    instance.add_thread(0, 0, 0);  // pid, tid, cid
    instance.set_task(task);
    // Create an Info
    husky::Info info = husky::utility::instance_to_info(instance, *obj->worker_info, {0, 0}, true);
    // Create a TableInfo
    TableInfo table_info {
        kv1, 9,
        husky::ModeType::SPMT, 
        husky::Consistency::None, 
        husky::WorkerType::None, 
        husky::ParamType::None
    };
    table_info.param_type = param_type;
    table_info.consistency = consistency;
    if (table_info.consistency == husky::Consistency::SSP) {
        table_info.kStaleness = 1;
    }

    // Test Push/Pull API
    {
        if (is_hogwild) {
            ml::mlworker::HogwildWorker<float> worker(info, table_info, *obj->zmq_context);
            testPushPull(worker, true);
        } else {
            ml::mlworker::SPMTWorker<float> worker(info, table_info, *obj->zmq_context);
            testPushPull(worker, true);
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // TODO, may have problem in SharedState, the destruction of socket may not release the bind address
    // Test V2 API
    {
        if (is_hogwild) {
            ml::mlworker::HogwildWorker<float> worker(info, table_info, *obj->zmq_context);
            testV2(worker, true);
        } else {
            ml::mlworker::SPMTWorker<float> worker(info, table_info, *obj->zmq_context);
            testV2(worker, true);
        }
    }
}

void test_multiple_threads(const husky::ParamType& param_type, const husky::Consistency& consistency, 
        TestSPMT* obj) {
    bool is_hogwild = false;
    if (consistency == husky::Consistency::None)
        is_hogwild = true;
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>("default_assign_map", -1, -1, 9, 2);
    // Create a task
    husky::MLTask task(0);
    // Create an Instance
    husky::Instance instance;
    instance.add_thread(0, 0, 0);  // pid, tid, cid
    instance.add_thread(0, 1, 1);  // pid, tid, cid
    instance.set_task(task);

    TableInfo table_info {
        kv1, 9,
        husky::ModeType::Single, 
        husky::Consistency::None, 
        husky::WorkerType::None, 
        husky::ParamType::None
    };
    table_info.consistency = consistency;
    table_info.param_type = param_type;
    if (table_info.consistency == husky::Consistency::SSP) {
        table_info.kStaleness = 1;
    }

    std::thread th1([&instance, obj, is_hogwild, table_info]() {
        if (is_hogwild) {
            // Create an Info
            husky::Info info = husky::utility::instance_to_info(instance, *obj->worker_info, {0, 0}, true);
            ml::mlworker::HogwildWorker<float> worker(info, table_info, *obj->zmq_context);

            testPushPull(worker, false);
            testV2(worker, false);
        } else {
            // Create an Info
            husky::Info info = husky::utility::instance_to_info(instance, *obj->worker_info, {0, 0}, true);
            ml::mlworker::SPMTWorker<float> worker(info, table_info, *obj->zmq_context);

            testPushPull(worker, false);
            testV2(worker, false);
        }
    });
    std::thread th2([&instance, obj, is_hogwild, table_info]() {
        if (is_hogwild) {
            // Create an Info
            husky::Info info = husky::utility::instance_to_info(instance, *obj->worker_info, {1, 1}, false);
            ml::mlworker::HogwildWorker<float> worker(info, table_info, *obj->zmq_context);

            testPushPull(worker, false);
            testV2(worker, false);

            std::this_thread::sleep_for(std::chrono::milliseconds(100)); 
            std::vector<float> vals;
            worker.Pull({1,3,5}, &vals);
            // for (int i = 0; i < vals.size(); ++ i)
            //     std::cout << vals[i] << " ";
            // std::cout << std::endl;
        } else {
            // Create an Info
            husky::Info info = husky::utility::instance_to_info(instance, *obj->worker_info, {1, 1}, false);
            ml::mlworker::SPMTWorker<float> worker(info, table_info, *obj->zmq_context);

            testPushPull(worker, false);
            testV2(worker, false);

            std::this_thread::sleep_for(std::chrono::milliseconds(100)); 
            std::vector<float> vals;
            worker.Pull({1,3,5}, &vals);
            // for (int i = 0; i < vals.size(); ++ i)
            //     std::cout << vals[i] << " ";
            // std::cout << std::endl;
        }
    });

    th1.join();
    th2.join();
}

TEST_F(TestSPMT, Single) {
    using namespace husky;
    for (auto param_type : {ParamType::IntegralType, ParamType::ChunkType}) {
        for (auto consistency_type : {Consistency::BSP, Consistency::SSP, Consistency::ASP, Consistency::None}) {
            test_single_thread(param_type, consistency_type, static_cast<TestSPMT*>(this));
        }
    }
}

TEST_F(TestSPMT, Multiple) {
    using namespace husky;
    for (auto param_type : {ParamType::IntegralType, ParamType::ChunkType}) {
        for (auto consistency_type : {Consistency::BSP, Consistency::SSP, Consistency::ASP, Consistency::None}) {
            test_multiple_threads(param_type, consistency_type, static_cast<TestSPMT*>(this));
        }
    }
}

}  // namespace
}  // namespace husky
