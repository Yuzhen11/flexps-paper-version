#include "gtest/gtest.h"

#include <iostream>
#include <thread>
#include <map>

#include "datastore/datastore.hpp"
#include "datastore/data_load_balance.hpp"

namespace datastore {
namespace {

class TestDataLoadBalance: public testing::Test {
   public:
    TestDataLoadBalance() {}
    ~TestDataLoadBalance() {}

   protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(TestDataLoadBalance, TestNext) {
    DataStore<int> data_store(3);  // datastore for 3 threads
    /*
     * data_map = {
     *  "0": [0, 3, 10, 13, 100, 102],   thread0 will get this set
     *  "1": [1, 4, 11, 14, 101, 104],   thread1 will get this set
     *  "2": [2, 12, 102]                thread2 will get this set
     * }
     */
    std::map<int, std::set<int>> data_map; 
    for (int i = 0; i < 5; ++ i) {
        data_store.Push(0, i);  // suppose thread 1 push i
        data_store.Push(1, i + 10);
        data_store.Push(2, i + 100);

        data_map[i % 3].insert(i);
        data_map[i % 3].insert(i + 10);
        data_map[i % 3].insert(i + 100);
    }

    std::thread t1([&data_map, &data_store]() {
        std::set<int> data_set1;
        DataLoadBalance<int> data_load_balance(data_store, 3, 0);
        data_load_balance.start_point();

        while(data_load_balance.has_next()) {
            data_set1.insert(data_load_balance.next());
        }
        EXPECT_EQ(data_map[0], data_set1);

    });

    std::thread t2([&data_map, &data_store]() {
        std::set<int> data_set2;
        DataLoadBalance<int> data_load_balance(data_store, 3, 1);
        data_load_balance.start_point();

        while(data_load_balance.has_next()) {
            data_set2.insert(data_load_balance.next());
        }
        EXPECT_EQ(data_map[1], data_set2);

    });

    std::thread t3([&data_map, &data_store]() {
        std::set<int> data_set3;
        DataLoadBalance<int> data_load_balance(data_store, 3, 2);
        data_load_balance.start_point();

        while(data_load_balance.has_next()) {
            data_set3.insert(data_load_balance.next());
        }
        EXPECT_EQ(data_map[2], data_set3);

    });

    t1.join();
    t2.join();
    t3.join();
}

}  // namespace
}  // namespace datastore
