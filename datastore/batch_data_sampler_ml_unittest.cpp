#include "gtest/gtest.h"

#include <chrono>
#include <vector>

#include <iostream>
#include <thread>
#include <map>

#include "husky/lib/ml/feature_label.hpp"

#include "datastore/datastore.hpp"
#include "datastore/batch_data_sampler.hpp"

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

namespace datastore {
namespace {

class TestBatchDataSampler: public testing::Test {
   public:
    TestBatchDataSampler() {}
    ~TestBatchDataSampler() {}

   protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(TestBatchDataSampler, TestNext) {
    datastore::DataStore<LabeledPointHObj<float, float, true>> data_store(2);
    std::set<float> data_set; 
    for (int i = 0; i < 5; ++ i) {
        LabeledPointHObj<float, float, true> obj1(i, 2 * i);
        LabeledPointHObj<float, float, true> obj2(i + 100, 2 * (i + 100));

        data_store.Push(0, obj1);  // suppose thread 1 push i
        data_store.Push(1, obj2);

        // insert the label
        data_set.insert(2 * i);
        data_set.insert(2 * (i + 100));
    }
    std::thread t1([&data_set, &data_store]() {
        std::set<float> data_set1;
        BatchDataSampler<LabeledPointHObj<float, float, true>> batch_data_sampler(data_store, 2);
        EXPECT_EQ(batch_data_sampler.empty(), false);
        batch_data_sampler.random_start_point();

        for (int i = 0; i < 5; i++) {
            auto keys = batch_data_sampler.prepare_next_batch();
            for (auto data : batch_data_sampler.get_data_ptrs()) {
                auto& x = data->x;
                float y = data->y;
                data_set1.insert(y);
            }
        } 
        
        EXPECT_EQ(data_set, data_set1);

    });

    std::thread t2([&data_set, &data_store]() {
        std::set<float> data_set2;
        BatchDataSampler<LabeledPointHObj<float, float, true>> batch_data_sampler(data_store, 5);
        EXPECT_EQ(batch_data_sampler.empty(), false);
        batch_data_sampler.random_start_point();

        for (int i = 0; i < 2; i++) {
            auto keys = batch_data_sampler.prepare_next_batch();
            for (auto data : batch_data_sampler.get_data_ptrs()) {
                auto& x = data->x;
                float y = data->y;
                data_set2.insert(y);
            }
        }

        EXPECT_EQ(data_set, data_set2);
    
    });

    t1.join();
    t2.join();
}

}  // namespace
}  // namespace datastore
