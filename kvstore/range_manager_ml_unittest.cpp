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
    void SetUp() {
        // At least create the RangeManager Singleton
        kvstore::RangeManager::Get();
    }
    void TearDown() {
        // Clear
        kvstore::RangeManager::Get().Clear();
    }
};

TEST_F(TestRangeManager, SetNumServers) {
    auto& range_manager = kvstore::RangeManager::Get();
    int num_servers = range_manager.GetNumServers();
    EXPECT_EQ(num_servers, -1);

    range_manager.SetNumServers(3);
    EXPECT_EQ(range_manager.GetNumServers(), 3);
}

TEST_F(TestRangeManager, GetNumRanges) {
    auto& range_manager = kvstore::RangeManager::Get();
    EXPECT_EQ(range_manager.GetNumRanges(), 0);

    range_manager.SetNumServers(3);
    range_manager.SetMaxKeyAndChunkSize(0, 9, 2);
    EXPECT_EQ(range_manager.GetNumRanges(), 1);

    range_manager.Clear();
    EXPECT_EQ(range_manager.GetNumRanges(), 0);
}

TEST_F(TestRangeManager, SetMaxKeyAndChunkSize) {
    auto& range_manager = kvstore::RangeManager::Get();
    range_manager.SetNumServers(3);
    int num_servers = range_manager.GetNumServers();
    EXPECT_EQ(num_servers, 3);
    // num_servers: 3, chunk_size: 2, max_key: 9
    // the result should be:
    // 5 chunks
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

TEST_F(TestRangeManager, GetServer) {
    // num_servers: 3, chunk_size: 2, max_key: 9
    // the result should be:
    // 5 chunks
    // {2, 2, 1}
    // {[0, 4), [4, 8), [8, 9)}
    auto& range_manager = kvstore::RangeManager::Get();
    range_manager.SetNumServers(3);
    range_manager.SetMaxKeyAndChunkSize(0, 9, 2);
    for (int i = 0; i < 4; ++ i)
        EXPECT_EQ(range_manager.GetServerFromKey(0, i), 0);
    for (int i = 4; i < 8; ++ i)
        EXPECT_EQ(range_manager.GetServerFromKey(0, i), 1);
    for (int i = 8; i < 9; ++ i)
        EXPECT_EQ(range_manager.GetServerFromKey(0, i), 2);

    EXPECT_EQ(range_manager.GetServerFromChunk(0, 0), 0);
    EXPECT_EQ(range_manager.GetServerFromChunk(0, 1), 0);
    EXPECT_EQ(range_manager.GetServerFromChunk(0, 2), 1);
    EXPECT_EQ(range_manager.GetServerFromChunk(0, 3), 1);
    EXPECT_EQ(range_manager.GetServerFromChunk(0, 4), 2);
}

TEST_F(TestRangeManager, CustomizeRanges) {
    // num_servers: 3, chunk_size: 2, max_key: 9
    // the result should be:
    // 5 chunks
    // {2, 2, 1}
    // {[0, 4), [4, 8), [8, 9)}
    auto& range_manager = kvstore::RangeManager::Get();
    range_manager.SetNumServers(3);
    int chunk_size = 2;
    int chunk_num = 5;
    husky::constants::Key max_key = 9;
    std::vector<kvstore::pslite::Range> server_key_ranges({{0, 4}, {4, 8}, {8, 9}});
    std::vector<kvstore::pslite::Range> server_chunk_ranges({{0, 2}, {2, 4}, {4, 5}});
    range_manager.CustomizeRanges(0, max_key, chunk_size, 
            chunk_num, server_key_ranges, server_chunk_ranges);

    for (int i = 0; i < 4; ++ i)
        EXPECT_EQ(range_manager.GetServerFromKey(0, i), 0);
    for (int i = 4; i < 8; ++ i)
        EXPECT_EQ(range_manager.GetServerFromKey(0, i), 1);
    for (int i = 8; i < 9; ++ i)
        EXPECT_EQ(range_manager.GetServerFromKey(0, i), 2);

    EXPECT_EQ(range_manager.GetServerFromChunk(0, 0), 0);
    EXPECT_EQ(range_manager.GetServerFromChunk(0, 1), 0);
    EXPECT_EQ(range_manager.GetServerFromChunk(0, 2), 1);
    EXPECT_EQ(range_manager.GetServerFromChunk(0, 3), 1);
    EXPECT_EQ(range_manager.GetServerFromChunk(0, 4), 2);
}

}  // namespace
}  // namespace husky
