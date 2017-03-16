#include "gtest/gtest.h"

#include "worker/model_transfer_store.hpp"
#include "worker/model_transfer_manager.hpp"

namespace husky {
namespace {

class TestModelTransferManager: public testing::Test {
   public:
    TestModelTransferManager() {}
    ~TestModelTransferManager() {}

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
    }
    void TearDown() {
        delete el;
        delete recver;
    }

    WorkerInfo worker_info;
    zmq::context_t zmq_context;
    MailboxEventLoop* el;
    CentralRecver * recver;
};

TEST_F(TestModelTransferManager, StartStop) {
    ModelTransferManager model_transfer_manager(worker_info, el, &zmq_context);
}

TEST_F(TestModelTransferManager, Send) {
    ModelTransferManager model_transfer_manager(worker_info, el, &zmq_context);
    // mailbox 0
    LocalMailbox mailbox0(&zmq_context);
    mailbox0.set_thread_id(0);
    el->register_mailbox(mailbox0);
    // mailbox 1
    LocalMailbox mailbox1(&zmq_context);
    mailbox1.set_thread_id(1);
    el->register_mailbox(mailbox1);

    // Add to ModelTransferStore
    std::vector<float> v{0.1, 0.2};
    husky::base::BinStream bin;
    bin << v;
    ModelTransferStore::Get().Add(0, std::move(bin));

    // send task
    model_transfer_manager.SendTask(1, 0);  // send to param 0 to 1

    // mailbox1 receives the params
    std::vector<float> params;
    if (mailbox1.poll(0,0)) {
        auto bin = mailbox1.recv(0,0);
        bin >> params;
        EXPECT_EQ(params[0], float(0.1));
        EXPECT_EQ(params[1], float(0.2));
    }
    ModelTransferStore::Get().Clear();
}

}  // namespace
}  // namespace husky
