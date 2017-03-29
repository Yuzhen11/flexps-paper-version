#include "examples/lda/lda_stat.hpp"

#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace husky {

const int LDAStats::kNumLogGammaAlpha_ = 100000;
const int LDAStats::kNumLogGammaAlphaSum_ = 10000;
const int LDAStats::kNumLogGammaBeta_ = 1000000;

void LDAStats::ComputeConstantVariables() {
    alpha_sum_ = K_ * alpha_;
    loggamma_alpha_offset_.resize(kNumLogGammaAlpha_);
    loggamma_alpha_sum_offset_.resize(kNumLogGammaAlphaSum_);
    loggamma_beta_offset_.resize(kNumLogGammaBeta_);
    for (int i = 0; i < kNumLogGammaAlpha_; ++i) {
        loggamma_alpha_offset_[i] = LogGamma(i + alpha_);
    }
    for (int i = 0; i < kNumLogGammaAlphaSum_; ++i) {
        loggamma_alpha_sum_offset_[i] = LogGamma(i + alpha_sum_);
    }
    for (int i = 0; i < kNumLogGammaBeta_; ++i) {
        loggamma_beta_offset_[i] = LogGamma(i + beta_);
    }
    // Precompute LLH parameters
    log_doc_normalizer_ = LogGamma(alpha_sum_) - K_ * LogGamma(alpha_);
}

LDAStats::LDAStats(const std::vector<int>& summary_table, int num_topics, int num_vocs, float alpha, float beta) : 
   summary_table_(summary_table), K_(num_topics), V_(num_vocs), alpha_(alpha), beta_(beta) {
    ComputeConstantVariables();
}

double LDAStats::GetLogGammaAlphaOffset(int val) {
    if (val < kNumLogGammaAlpha_) {
        return loggamma_alpha_offset_[val];
    }
    return LogGamma(val + alpha_);
}

double LDAStats::GetLogGammaAlphaSumOffset(int val) {
    if (val < kNumLogGammaAlphaSum_) {
        return loggamma_alpha_sum_offset_[val];
    }
    return LogGamma(val + alpha_sum_);
}

double LDAStats::GetLogGammaBetaOffset(int val) {
    if (val < kNumLogGammaBeta_) {
        return loggamma_beta_offset_[val];
    }
    return LogGamma(val + beta_);
}

double LDAStats::ComputeOneDocLLH(LDADoc& doc) {
    double one_doc_llh = log_doc_normalizer_;
    // Compute doc-topic vector on the fly.
    std::vector<int> doc_topic_vec(K_);
    std::fill(doc_topic_vec.begin(), doc_topic_vec.end(), 0);
    int num_words = 0;
    for (LDADoc::Iterator it(&doc); !it.IsEnd(); it.Next()) {
        ++doc_topic_vec[it.Topic()];
        ++num_words;
    }
    for (int k = 0; k < K_; ++k) {
        assert(doc_topic_vec[k] >= 0);
        one_doc_llh += GetLogGammaAlphaOffset(doc_topic_vec[k]);
    }
    one_doc_llh -= GetLogGammaAlphaSumOffset(num_words);
    assert(one_doc_llh <= 0);
    assert(!isinf(one_doc_llh));
    return one_doc_llh;
}

double LDAStats::ComputeWordLLH(int word_idx_start, int word_idx_end, std::vector<int>& word_topic_table) {
    double word_llh = 0.;
    static double zero_entry_llh = GetLogGammaBetaOffset(0);
    int num_words = word_idx_end - word_idx_start;
    word_idx_start = 0;
    word_idx_end = num_words;
    for (int w = word_idx_start; w < word_idx_end; ++w) {
        int num_zeros = 0;
        for (int i=0; i<K_; i++) {
            int count = word_topic_table[w * K_ + i];
            if (count == 0) {
                num_zeros++;
                continue;
            }
            word_llh += GetLogGammaBetaOffset(count);
        }   
        // The other word-topic counts are 0.
        if (num_zeros < K_) {
            word_llh += num_zeros * zero_entry_llh;
        }
    }
    assert(!isinf(word_llh));
    return word_llh;
}


double LDAStats::ComputeWordLLHSummary() {
    beta_sum_ = beta_ * V_;
    log_topic_normalizer_ = K_ * (LogGamma(beta_sum_) - V_ * LogGamma(beta_));
    double word_llh = log_topic_normalizer_;
    // log(\prod_j (1 / \gamma(n_j^* + W\beta))) term.
    for (int k = 0; k < K_; ++k) {
        assert(!isinf(word_llh));
        word_llh -= LogGamma(summary_table_[k] + beta_sum_);
    }
    return word_llh;
}

}  // namespace husky
