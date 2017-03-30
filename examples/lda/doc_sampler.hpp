#pragma once
#include "examples/lda/lda_doc.hpp"
#include "kvstore/kvworker.hpp"

namespace husky {

// DocSampler does not store any documents.
class DocSampler {
   public:   
    DocSampler(int num_topics, int num_vocs, float alpha, float beta); 
    ~DocSampler(); 

    void sample_one_doc(LDADoc& doc, std::vector<int>& word_topic, std::vector<int>& word_topic_updates, std::map<int, int>& find_start_idx);

    void sample_one_doc(LDADoc& doc, std::vector<std::vector<int>>& word_topic, std::vector<std::vector<int>>& word_topic_updates, std::map<int, int>& find_start_idx);

    // Compute the following fast sampler auxiliary variables:
    //
    // 1. mass in s bucket and store in s_sum_.
    // 2. mass in r bucket and store in r_sum_.
    // 3. coefficients in the q term and store in q_coeff_.
    // 4. populate nonzero_doc_topic_idx_ based on theta_.
    //
    // This requires that theta_ is already computed.
    void compute_aux_variables();

   private:
    void compute_doc_topic_vector(LDADoc& doc);

    int sample();

   private:  // private members.
    // ================ Topic Model Parameters =================
    // Number of topics.
    int K_;

    // Number of vocabs.
    int V_;

    // Dirichlet prior for word-topic vectors.
    double beta_;

    // Equals to V_*beta_
    double beta_sum_;

    // Dirichlet prior for doc-topic vectors.
    double alpha_;

    std::vector<int> topic_summary_;

    // ============== Fast Sampler (Cached) Variables ================

    // The mass of s bucket in the paper, recomputed before sampling a document.
    //
    // Comment(wdai): In principle this is independent of a document, and only
    // needs to be recomputed when summary row is refreshed. However, since it's
    // cumbersome interface to let the sampler know when summary row is
    // refreshed, we recompute this at the beginning of each document (very
    // cheap).
    //
    // Comment(wdai): We do not maintain each term in the sum because for small
    // alpha_ and beta_, the chance of a sample falling into this bucket is so
    // small (<10%) that it might be cheaper to compute the necessary terms on
    // the fly when this bucket is picked instead of maintaining the vector of
    // terms.
    double s_sum_;

    // The mass of r bucket in the paper. This needs to be computed for each
    // document, and updated incrementally as each token in the doc is sampled.
    //
    // Comment(wdai): We don't maintain each terms in the sum for the same
    // reasons as s_sum_.
    double r_sum_;

    // The mass of q bucket in the paper. This is computed for each word with
    // the help of q_coeff_.
    double q_sum_;

    // The cached coefficients in the q bucket. The coefficients are the terms
    // without word-topic count. [K_ x 1] dimension. This is computed before
    // sampling of each document, and incrementally updated.
    std::vector<double> q_coeff_;

    // Nonzero terms in the summand of q bucket. [K_ x 1] dimension, but only
    // the first num_nonzero_q_terms_ enetries are active.
    std::vector<double> nonzero_q_terms_;

    // The corresponding topic index in nonzero_q_terms_. [K_ x 1] dimension.
    std::vector<double> nonzero_q_terms_topic_;

    // q_terms_ is sparse as word-topic count is sparse (0 for most topics).
    int num_nonzero_q_terms_;

    // Compute theta_ before sampling a document. This remove the need
    // to store the potentially big doc-topic vector in each document, and
    // shouldn't add much runtime overhead.
    //
    // Comment(wdai): theta_ is sparse with large K_. If we use
    // memory-efficient way, e.g. map, to represent it the access time would be
    // high.
    std::vector<int> theta_;

    // Sorted indices of nonzero topics in theta_ to exploit sparsity in
    // doc_topic_vec. Sorted so we can use std::lower_bound binary search. This
    // has size [K x 1], but only the first num_nonzero_idx_ entries are active.
    //
    // Comment(wdai): Use POD array to allow memmove, which is faster than using
    // std::vector with std::move.
    int* nonzero_doc_topic_idx_;

    // Number of non-zero entries in theta_.
    int num_nonzero_doc_topic_idx_;

    // ================== Utilities ====================
    std::unique_ptr<std::mt19937> rng_engine_;

    std::uniform_real_distribution<double> uniform_zero_one_dist_;
};

}  // namespace husky
