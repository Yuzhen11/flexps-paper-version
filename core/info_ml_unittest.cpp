#include "gtest/gtest.h"

#include <iostream>
#include <thread>
#include <set>

#include "core/info.hpp"

namespace husky {
namespace {

class TestInfo: public testing::Test {
   public:
    TestInfo() {}
    ~TestInfo() {}

   protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(TestInfo, TestNext) {
    Info info;
    // set info
    info.set_local_id(1);
    info.set_global_id(2);
    info.set_cluster_id(3);
    info.set_current_epoch(5);

    EXPECT_EQ(info.get_local_id(), 1);
    EXPECT_EQ(info.get_global_id(), 2);
    EXPECT_EQ(info.get_cluster_id(), 3);
    EXPECT_EQ(info.get_current_epoch(), 5);
}

}  // namespace
}  // namespace datastore
