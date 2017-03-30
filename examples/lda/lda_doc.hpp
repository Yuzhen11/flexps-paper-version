#pragma once

#include "husky/base/log.hpp"

#include <random>
#include <string>
#include <sstream>
#include <vector>

namespace husky {

class LDADoc {
   public:
    LDADoc() : num_topics_(0) {}
    ~LDADoc() {}
  
    // Initialize the topic assignments to -1.
    void append_word(int word_id, size_t count) {
        for (int i = 0; i < count; ++i) {
            tokens_.push_back(word_id);
            token_topics_.push_back(-1);
        }
    }
  
    inline int get_token(int idx) const {
        return tokens_[idx];
    }
  
    inline int get_num_tokens() const {
        return tokens_.size();
    }
  
    inline int get_topic(int idx) const {
        return token_topics_[idx];
    }
  
    inline int get_num_topics() const {
        return num_topics_;
    }
  
    // Print doc as "word:topic" pairs.
    std::string print_doc() {
        std::stringstream ss;
        for (LDADoc::Iterator it(this); !it.IsEnd(); it.Next()) {
            ss << it.Word() << ":" << it.Topic() << " ";
        }
        return ss.str();
    }
  
    void random_init_topics(int num_topics) {
        num_topics_ = num_topics;
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> one_K_rng(0, num_topics - 1);
        for (int i = 0; i < tokens_.size(); ++i) {
            token_topics_[i] = one_K_rng(gen);
        }
    }
  
  public:   // Iterator
    // Iterator lets you do this:
    //
    //  LDADoc doc;
    //  // ... Initialize it in some way ...
    //  for (LDADoc::Iterator it(&doc); !it.IsEnd(); it.Next()) {
    //    int word = it.Word();
    //    int topic = it.Topic();
    //    it.SetTopic(0);
    //  }
    //
    // Notice the is_end() in for loop. You can't use the == operator.
    class Iterator {
       public:
        // Does not take the ownership of doc.
        Iterator(LDADoc* doc) : doc_(*doc), curr_(0) { }
  
        inline int Word() {
            return doc_.tokens_[curr_];
        }
  
        inline int Topic() {
            return doc_.token_topics_[curr_];
        }
  
        inline void Next() {
            ++curr_;
        }
  
        inline bool IsEnd() {
            return curr_ == doc_.tokens_.size();
        }
  
        inline void SetTopic(int new_topic) {
            assert(curr_< doc_.token_topics_.size());
            doc_.token_topics_[curr_] = new_topic;
        }
  
       private:
        LDADoc& doc_;

        // The index to the tokens_ and token_topics_.
        int curr_;
    };

   private:
    // tokens_[i] has topic token_topics_[i].
    std::vector<int> tokens_;
    std::vector<int> token_topics_;
    int num_topics_;
};

} // namespace husky
