#include "gtest/gtest.h"

#include "worker/model_transfer_store.hpp"

namespace husky {
namespace {

class TestModelTransferStore: public testing::Test {
   public:
    TestModelTransferStore() {}
    ~TestModelTransferStore() {}

   protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(TestModelTransferStore, StartStop) {
    auto& store = ModelTransferStore::Get();
    store.Clear();
}

TEST_F(TestModelTransferStore, AddPop) {
    auto& store = ModelTransferStore::Get();
    // Add
    store.Add(0, {0.1, 0.2});
    EXPECT_EQ(store.Size(), 1);
    store.Add(1, {0.3, 0.4});
    EXPECT_EQ(store.Size(), 2);

    // Pop
    std::vector<float> params = store.Pop(0);
    EXPECT_EQ(params[0], float(0.1));
    EXPECT_EQ(params[1], float(0.2));
    EXPECT_EQ(store.Size(), 1);
    params = store.Pop(1);
    EXPECT_EQ(params[0], float(0.3));
    EXPECT_EQ(params[1], float(0.4));
    EXPECT_EQ(store.Size(), 0);
    store.Clear();
}

}  // namespace
}  // namespace husky
