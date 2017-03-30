#include "examples/lda/lda_doc.hpp"

#include <set>
#include <unordered_map>

#include "gtest/gtest.h"
#include "husky/base/log.hpp"

namespace husky {
namespace {

class TestLDADoc : public testing::Test {
   public:
    TestLDADoc() {}
    ~TestLDADoc() {}
   protected:
    void SetUp() {}
    void TearDown() {}
};


TEST_F(TestLDADoc,  TestAppendWord) {
    LDADoc doc;
    for (int i = 0; i < 5; ++i) {
        doc.append_word(i, 2); // every word appears twice
    }
    EXPECT_EQ(doc.get_num_topics(), 0);
    EXPECT_EQ(doc.get_num_tokens(), 10);
}

TEST_F(TestLDADoc,  TestRandomInitTopics) {
    LDADoc doc;
    for (int i = 0; i < 5; ++i) {
        doc.append_word(i, 2); // every word appears twice
    }
    doc.random_init_topics(5);
    EXPECT_EQ(doc.get_num_topics(), 5);
}

TEST_F(TestLDADoc,  TestIterator) {
    LDADoc doc;
    for (int i = 0; i < 5; ++i) {
        doc.append_word(i, 2); // every word appears twice
    }
    for (LDADoc::Iterator it(&doc); !it.IsEnd(); it.Next()) {
      int word = it.Word();
      int topic = it.Topic();
      it.SetTopic(0);
    }
    for (int i = 0; i < doc.get_num_tokens(); ++i) {
        EXPECT_EQ(doc.get_topic(i), 0);
    }
}

} // namespace annoymous
} // namespace husky
