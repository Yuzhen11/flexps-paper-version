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
    std::vector<float> v{0.1, 0.2};
    husky::base::BinStream bin;
    bin << v;
    store.Add(0, std::move(bin));
    EXPECT_EQ(store.Size(), 1);
    std::vector<float> v2{0.3, 0.4};
    husky::base::BinStream bin2;
    bin2 << v2;
    store.Add(1, std::move(bin2));
    EXPECT_EQ(store.Size(), 2);

    // Pop
    auto bin3 = store.Pop(0);
    std::vector<float> params;
    bin3 >> params;
    EXPECT_EQ(params[0], float(0.1));
    EXPECT_EQ(params[1], float(0.2));
    EXPECT_EQ(store.Size(), 1);
    bin3 = store.Pop(1);
    bin3 >> params;
    EXPECT_EQ(params[0], float(0.3));
    EXPECT_EQ(params[1], float(0.4));
    EXPECT_EQ(store.Size(), 0);
    store.Clear();
}

}  // namespace
}  // namespace husky
