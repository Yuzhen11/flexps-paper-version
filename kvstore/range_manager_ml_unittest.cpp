#include <iostream>

#include "range_manager.hpp"

#include "gtest/gtest.h"

namespace husky {
namespace {

class TestRangeManager: public testing::Test {
   public:
    TestRangeManager() {}
    ~TestRangeManager() {}

   protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(TestRangeManager, SetNumServers) {
    auto& range_manager = kvstore::RangeManager::Get();
    int num_servers = range_manager.GetNumServers();
    EXPECT_EQ(num_servers, -1);
}

TEST_F(TestRangeManager, SetMaxKeyAndChunkSize) {
    auto& range_manager = kvstore::RangeManager::Get();
    range_manager.SetNumServers(3);
    int num_servers = range_manager.GetNumServers();
    EXPECT_EQ(num_servers, 3);
    // num_servers: 3, chunk_size: 2, max_key: 9
    // the result should be:
    // {2, 2, 1}
    // {[0, 4), [4, 8), [8, 9)}
    range_manager.SetMaxKeyAndChunkSize(0, 9, 2);
    auto ranges = range_manager.GetServerKeyRanges(0);
    EXPECT_EQ(ranges[0].begin(), 0);
    EXPECT_EQ(ranges[0].end(), 4);
    EXPECT_EQ(ranges[1].begin(), 4);
    EXPECT_EQ(ranges[1].end(), 8);
    EXPECT_EQ(ranges[2].begin(), 8);
    EXPECT_EQ(ranges[2].end(), 9);
    auto chunk_ranges = range_manager.GetServerChunkRanges(0);
    EXPECT_EQ(chunk_ranges[0].begin(), 0);
    EXPECT_EQ(chunk_ranges[0].end(), 2);
    EXPECT_EQ(chunk_ranges[1].begin(), 2);
    EXPECT_EQ(chunk_ranges[1].end(), 4);
    EXPECT_EQ(chunk_ranges[2].begin(), 4);
    EXPECT_EQ(chunk_ranges[2].end(), 5);
}

}  // namespace
}  // namespace husky
