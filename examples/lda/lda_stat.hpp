#pragma once

#include "examples/lda/lda_doc.hpp"
#include <string>
#include <vector>

namespace husky {


// We reference Griffiths and Steyvers (PNAS, 2003).
class LDAStats {
   public:
    LDAStats(const std::vector<int>& summary_table, int num_topics, int num_vocs, float alpha, float beta);

    // This is part of log P(z) in eq.[3].
    double ComputeOneDocLLH(LDADoc& doc);

    // This computes the terms in numerator part of log P(w|z) in eq.[2].
    // Covers words within [word_idx_start, word_idx_end).
    double ComputeWordLLH(int word_idx_start, int word_idx_end, std::vector<int>& word_topic_table);

    // Only 1 client should call this in a iteration.
    double ComputeWordLLHSummary();

   private:  // private functions
    // Get (from cache or computed afresh) loggamma(val + alpha_).
    double GetLogGammaAlphaOffset(int val);

    // Get (from cache or computed afresh) loggamma(val + alpha_sum_).
    double GetLogGammaAlphaSumOffset(int val);

    // Get (from cache or computed afresh) loggamma(val + beta_).
    double GetLogGammaBetaOffset(int val);

    double LogGamma(double xx) {
      assert(xx !=0 );
      int j;
      double x, y, tmp1, ser;
      y = xx;
      x = xx;
      tmp1 = x+5.5;
      tmp1 -= (x+0.5)*log(tmp1);
      ser = 1.000000000190015;
      for (j = 0; j < 6; j++) ser += cof_[j]/++y;
      return -tmp1+log(2.5066282746310005*ser/x);
    }

    void ComputeConstantVariables();

   private:
    // ================ Topic Model Parameters =================
    // Number of topics.
    int K_;

    // Number of vocabs.
    int V_;

    // Dirichlet prior for word-topic vectors.
    float beta_;

    // V_*beta_
    float beta_sum_;

    // Dirichlet prior for doc-topic vectors.
    float alpha_;

    // K_*alpha__
    float alpha_sum_;

    // Used in LogGamma
    const double cof_[6] = { 76.18009172947146, -86.50532032941677,
                       24.01409824083091, -1.231739572450155,
                       0.1208650973866179e-2, -0.5395239384953e-5
                     };

    // ================ Precompute/Cached LLH Parameters =================
    // Log of normalization constant (per docoument) from eq.[3].
    double log_doc_normalizer_;

    // Log of normalization constant (per topic) from eq.[2].
    double log_topic_normalizer_;

    // Pre-compute loggamma.
    std::vector<double> loggamma_alpha_offset_;
    std::vector<double> loggamma_alpha_sum_offset_;
    std::vector<double> loggamma_beta_offset_;

    // Number of caches.
    // About # of tokens in a topic in a doc.
    static const int kNumLogGammaAlpha_;
    // About # of words in a doc.
    static const int kNumLogGammaAlphaSum_;
    // About # of tokens in a topic.
    static const int kNumLogGammaBeta_;

    // ============== Global Parameters from Petuum Server =================
    // A table containing just one summary row of [K x 1] dimension. The k-th
    // entry in the table is the # of tokens assigned to topic k.
    const std::vector<int>& summary_table_;

};

}  // namespace husky
