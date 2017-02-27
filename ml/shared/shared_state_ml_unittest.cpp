#include "gtest/gtest.h"

#include <thread>

#include "ml/shared/shared_state.hpp"

namespace ml {
namespace {

class TestSharedState : public testing::Test {
   public:
    TestSharedState() {}
    ~TestSharedState() {}

   protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(TestSharedState, Create) {
    zmq::context_t context;
    SharedState<int> s(0, 0, 1, context);
}

TEST_F(TestSharedState, Init) {
    zmq::context_t context;
    std::vector<std::thread> ths;
    int num_threads = 2;
    for (int i = 0; i < num_threads; ++ i) {
        ths.push_back(std::thread([&context, i, num_threads]() {
            SharedState<int> s(0, i, num_threads, context);
            if (i == 0) {
                int* p = new int;
                s.Init(p);
            }
            if (i == 0) {
                delete s.Get();
            }
        }));
    }
    for (auto& th : ths) 
        th.join();
}

TEST_F(TestSharedState, SyncState) {
    zmq::context_t context;
    std::vector<std::thread> ths;
    int num_threads = 2;
    for (int i = 0; i < num_threads; ++ i) {
        ths.push_back(std::thread([&context, i, num_threads]() {
            SharedState<int> s(0, i, num_threads, context);
            if (i == 0) {
                int* p = new int;
                *p = 10;
                s.Init(p);
            }
            s.SyncState();
            EXPECT_EQ(*s.Get(), 10);
            s.Barrier();
            if (i == 0) {
                delete s.Get();
            }
        }));
    }
    for (auto& th : ths) 
        th.join();
}

TEST_F(TestSharedState, Barrier) {
    zmq::context_t context;
    std::vector<std::thread> ths;
    int num_threads = 2;
    for (int i = 0; i < num_threads; ++ i) {
        ths.push_back(std::thread([&context, i, num_threads]() {
            SharedState<int> s(0, i, num_threads, context);
            if (i == 0) {
                int* p = new int;
                *p = 10;
                s.Init(p);
            }
            s.Barrier();
            s.Barrier();
            s.Barrier();
            if (i == 0) {
                delete s.Get();
            }
        }));
    }
    for (auto& th : ths) 
        th.join();
}

}  // namespace
}  // namespace ml
