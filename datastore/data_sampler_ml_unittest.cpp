#include "gtest/gtest.h"

#include <iostream>
#include <thread>
#include <set>

#include "datastore/data_sampler.hpp"

namespace datastore {
namespace {

class TestDataSampler: public testing::Test {
   public:
    TestDataSampler() {}
    ~TestDataSampler() {}

   protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(TestDataSampler, TestNext) {
    DataStore<int> data_store(2);  // datastore for 2 threads
    std::set<int> data_set;
    for (int i = 0; i < 5; ++ i) {
        data_store.Push(0, i);  // suppose thread 1 push i
        data_set.insert(i);
        data_store.Push(1, i+10);  // suppose thread 2 push i+10
        data_set.insert(i+10);
    }
    std::thread t1([&data_set, &data_store](){
        std::set<int> data_set2;
        DataSampler<int> data_sampler(data_store);
        data_sampler.random_start_point();
        for (int i = 0; i < 10; ++ i) {  // read 10 numbers, it should go through everything
            int data = data_sampler.next();
            data_set2.insert(data);
        }
        EXPECT_EQ(data_set, data_set2);
    });
    t1.join();
}

}  // namespace
}  // namespace datastore
