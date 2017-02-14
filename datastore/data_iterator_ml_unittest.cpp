#include "gtest/gtest.h"

#include <iostream>

#include "datastore/data_iterator.hpp"

namespace datastore {
namespace {

class TestDataIterator: public testing::Test {
   public:
    TestDataIterator() {}
    ~TestDataIterator() {}

   protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(TestDataIterator, TestNext) {
    DataStore<int> data_store(2);  // datastore for 2 threads
    std::set<int> data_set;
    for (int i = 0; i < 5; ++ i) {
        data_store.Push(0, i);  // suppose thread 1 push i
        data_set.insert(i);
        data_store.Push(1, i+10);  // suppose thread 2 push i+10
        data_set.insert(i+10);
    }

    std::set<int> data_set2;
    DataIterator<int> data_iterator(data_store);
    while (data_iterator.has_next()) {  // has_next
        int data = data_iterator.next();  // next
        data_set2.insert(data);
    }
    EXPECT_EQ(data_set, data_set2);
}

}  // namespace
}  // namespace datastore
