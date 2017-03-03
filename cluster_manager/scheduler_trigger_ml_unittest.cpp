#include "cluster_manager/scheduler_trigger.hpp"

#include <vector>
#include <memory>
#include <iostream>
#include <thread>
#include <chrono>

#include "gtest/gtest.h"
#include "zmq.hpp"

#include "husky/core/zmq_helpers.hpp"
#include "core/constants.hpp"

namespace husky {
namespace {

// Mock the logic of cluster manager
class MockClusterManager {
   public:
    MockClusterManager(zmq::context_t* con) : context(con), cluster_manager_addr("inproc://TestSchedulerTrigger") {}
    ~MockClusterManager() {
        delete server;
    }
    void init() {
        server = new zmq::socket_t(*context, ZMQ_PULL);    
        server->bind(cluster_manager_addr);
        st.reset(new SchedulerTrigger(context, cluster_manager_addr));
    }
    void serve() {
        while (true) {
            int type = zmq_recv_int32(server);
            if (type == constants::kClusterManagerTimeOutSchedule) {
                int ts = zmq_recv_int32(server);
                if (st->is_current_ts(ts)) {
                    std::cout<<"Start timeout scheduling\n";
                } else {
                    std::cout<<"Old timestamp skipped\n";
                }
            } else if (type == constants::kClusterManagerThreadFinished) {
                if (st->has_enough_new_threads()) {
                    std::cout<<"Start kthreadfinished scheduling\n";
                }
            } else if (type == constants::kClusterManagerExit) {
                break;
            }
        }
    }

    std::string cluster_manager_addr;
    zmq::context_t* context;
    zmq::socket_t* server;
    std::unique_ptr<SchedulerTrigger> st;
};

class TestSchedulerTrigger :  public testing::Test {
   public:
    TestSchedulerTrigger() {}

    ~TestSchedulerTrigger() {}
   protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(TestSchedulerTrigger, TestConstruct) {
    zmq::context_t CONTEXT;
    MockClusterManager mcm(&CONTEXT);
    mcm.init();

    // default timeout period is 5 seconds
    // default threads threshold is 1
    EXPECT_EQ(5, mcm.st->get_time_out_period());
    EXPECT_EQ(1, mcm.st->get_count_threshold());
    mcm.st->set_time_out_period(1);
    mcm.st->set_count_threshold(2);
    EXPECT_EQ(1, mcm.st->get_time_out_period());
    EXPECT_EQ(2, mcm.st->get_count_threshold());
}

TEST_F(TestSchedulerTrigger, TestBothStrategy) {
    zmq::context_t CONTEXT;
    MockClusterManager mcm(&CONTEXT);
    mcm.init();
    zmq::socket_t event_sender(CONTEXT, ZMQ_PUSH);
    event_sender.connect(mcm.cluster_manager_addr);

    std::thread* mcm_thread = new std::thread([&mcm]() {
            mcm.serve();
    });

    std::thread* event_sender_thread = new std::thread([&mcm, &CONTEXT, &event_sender]() {
        // Set the timout period to be 
        mcm.st->set_time_out_period(1);
        mcm.st->set_count_threshold(10); 

        // Try to generate time_out schedule event
        for (int i = 0; i < 5; i++)
            zmq_send_int32(&event_sender, constants::kClusterManagerThreadFinished);
        std::this_thread::sleep_for(std::chrono::seconds(2)); // larger than timeout period

        // Try to generate kthreadfinished event
        for (int i = 0; i < 9; i++)
            zmq_sendmore_int32(&event_sender, constants::kClusterManagerThreadFinished);
        zmq_send_int32(&event_sender, constants::kClusterManagerThreadFinished);
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Try to generate kthreadfinished event first then timeout event
        for (int i = 0; i < 15; i++)
            zmq_send_int32(&event_sender, constants::kClusterManagerThreadFinished);
        std::this_thread::sleep_for(std::chrono::seconds(2));

        zmq_send_int32(&event_sender, constants::kClusterManagerExit);
    });
    
    mcm_thread->join();
    event_sender_thread->join();

    delete mcm_thread;
    delete event_sender_thread;
}

} // namespace 
} // namespace husky
