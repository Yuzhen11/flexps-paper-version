#include "examples/lda/doc_sampler.hpp"

#include <vector>
#include <random>
#include <sstream>
#include <string>
#include <iostream>

namespace husky {

DocSampler::DocSampler(int num_topics, int num_vocs, float alpha, float beta)
    : K_(num_topics), V_(num_vocs), alpha_(alpha), beta_(beta) {

    std::random_device rd;
    rng_engine_.reset(new std::mt19937(rd()));

    topic_summary_.resize(num_topics);

    // Fast sampler variables
    beta_sum_ = num_vocs * beta;
    theta_.resize(num_topics);
    q_coeff_.resize(num_topics);
    nonzero_q_terms_.resize(num_topics);
    nonzero_q_terms_topic_.resize(num_topics);
    nonzero_doc_topic_idx_ = new int[num_topics];
}

 
DocSampler::~DocSampler() {
    delete[] nonzero_doc_topic_idx_;
}

void DocSampler::sample_one_doc(LDADoc& doc, std::vector<std::vector<int>>& word_topic_count, std::vector<std::vector<int>>& word_topic_updates, std::map<int, int>& find_start_idx) { 
    // clear previous topic_summary_
    std::fill(topic_summary_.begin(), topic_summary_.end(), 0);
    int topic_summary_start  = find_start_idx[V_];
    assert(topic_summary_start!=0);
    topic_summary_ = word_topic_count[topic_summary_start]; 

    // Preparations before sampling a document.  
    // Compute the specific aux variables for this doc.
    compute_doc_topic_vector(doc); 
    compute_aux_variables();

    for (LDADoc::Iterator it(&doc); !it.IsEnd(); it.Next()) {
        int old_topic = it.Topic();
        int word = it.Word();
        int word_start = find_start_idx[word];

        // Remove this token from corpus. It requires following updates:
        // 1. Update terms in s, r and q bucket.
        double denom = topic_summary_[old_topic] + beta_sum_;
        s_sum_ -= (alpha_ * beta_) / denom;
        s_sum_ += (alpha_ * beta_) / (denom - 1);
        r_sum_ -= (theta_[old_topic] * beta_) / denom;
        r_sum_ += ((theta_[old_topic] - 1) * beta_) / (denom - 1);
        q_coeff_[old_topic] =
          (alpha_ + theta_[old_topic] - 1) / (denom - 1);

        // 2. theta_. If old_topic token count goes to 0, update the
        // nonzero_doc_topic_idx_, etc.
        --theta_[old_topic];
        if (theta_[old_topic] == 0) {
            int* zero_idx =
              std::lower_bound(nonzero_doc_topic_idx_,
                  nonzero_doc_topic_idx_ + num_nonzero_doc_topic_idx_, old_topic);
            memmove(zero_idx, zero_idx + 1,
                (nonzero_doc_topic_idx_ + num_nonzero_doc_topic_idx_ - zero_idx - 1)
                * sizeof(int));
            --num_nonzero_doc_topic_idx_;
        }

        // 3. directly modify the local cache and updates
        topic_summary_[old_topic] -= 1;
        word_topic_count[word_start][old_topic] -= 1;
        word_topic_updates[topic_summary_start][old_topic] -= 1;
        word_topic_updates[word_start][old_topic] -= 1;

        // 4. Compute the mass in q bucket by iterating through the sparse word-topic
        // vector.
        q_sum_ = 0.;
        // Go through the list of word topic row with word id  equal to it.Word()
        // Update accordingly
        num_nonzero_q_terms_ = 0;
        for (int topic=0; topic<K_; ++topic) {
            int count = word_topic_count[word_start][topic];
            // skip zero count terms
            if (count > 0) {
                nonzero_q_terms_[num_nonzero_q_terms_] = (topic == old_topic) ?
                  (q_coeff_[topic] * (count - 1)) : (q_coeff_[topic] * count);
                nonzero_q_terms_topic_[num_nonzero_q_terms_] = topic;
                q_sum_ += nonzero_q_terms_[num_nonzero_q_terms_];
                ++num_nonzero_q_terms_;
            }
        }

        // Sample the topic for token 'it' using the aux variables.
        int new_topic = sample();
        assert(new_topic<K_);

        // 5. Add this token with new topic back to corpus using following steps:
        //
        // 5.1 Update s, r and q bucket.
        denom = topic_summary_[new_topic] + beta_sum_;
        s_sum_ -= (alpha_ * beta_) / denom;
        s_sum_ += (alpha_ * beta_) / (denom + 1);
        r_sum_ -= (theta_[new_topic] * beta_) / denom;
        r_sum_ += ((theta_[new_topic] + 1) * beta_) / (denom + 1);
        q_coeff_[new_topic] =
          (alpha_ + theta_[new_topic] + 1) / (denom + 1);


        // 5.2 theta_. If new_topic token count goes to 1, we need to add it
        // to the nonzero_doc_topic_idx_, etc.
        ++theta_[new_topic];   
        if (theta_[new_topic] == 1) {
            int* insert_idx =
              std::lower_bound(nonzero_doc_topic_idx_,
                  nonzero_doc_topic_idx_ + num_nonzero_doc_topic_idx_, new_topic);
            memmove(insert_idx + 1, insert_idx, (nonzero_doc_topic_idx_ +
                  num_nonzero_doc_topic_idx_ - insert_idx) * sizeof(int));
            *insert_idx = new_topic;
            ++num_nonzero_doc_topic_idx_;
        }

        // directly modify cache and updates
        topic_summary_[new_topic] += 1;
        word_topic_count[word_start][new_topic] += 1;
        word_topic_updates[topic_summary_start][new_topic] += 1;
        word_topic_updates[word_start][new_topic] += 1;

        // Finally, update the topic assignment z in doc
        if (old_topic != new_topic) {
            it.SetTopic(new_topic);
        }
    } // end of doc itration
}

void DocSampler::sample_one_doc(LDADoc& doc, std::vector<int>& word_topic_count, std::vector<int>& word_topic_updates, std::map<int, int>& find_start_idx) { 
    // clear previous topic_summary_
    std::fill(topic_summary_.begin(), topic_summary_.end(), 0);
    int topic_summary_start  = find_start_idx[V_];
    for (int k = 0; k < K_; k++) {
        topic_summary_[k] = word_topic_count[topic_summary_start + k];
    }

    // Preparations before sampling a document.  
    // Compute the specific aux variables for this doc.
    compute_doc_topic_vector(doc); 
    compute_aux_variables();

    for (LDADoc::Iterator it(&doc); !it.IsEnd(); it.Next()) {
        int old_topic = it.Topic();
        int word = it.Word();
        int word_start = find_start_idx[word];

        // Remove this token from corpus. It requires following updates:
        // 1. Update terms in s, r and q bucket.
        double denom = topic_summary_[old_topic] + beta_sum_;
        s_sum_ -= (alpha_ * beta_) / denom;
        s_sum_ += (alpha_ * beta_) / (denom - 1);
        r_sum_ -= (theta_[old_topic] * beta_) / denom;
        r_sum_ += ((theta_[old_topic] - 1) * beta_) / (denom - 1);
        q_coeff_[old_topic] =
          (alpha_ + theta_[old_topic] - 1) / (denom - 1);

        // 2. theta_. If old_topic token count goes to 0, update the
        // nonzero_doc_topic_idx_, etc.
        --theta_[old_topic];
        if (theta_[old_topic] == 0) {
            int* zero_idx =
              std::lower_bound(nonzero_doc_topic_idx_,
                  nonzero_doc_topic_idx_ + num_nonzero_doc_topic_idx_, old_topic);
            memmove(zero_idx, zero_idx + 1,
                (nonzero_doc_topic_idx_ + num_nonzero_doc_topic_idx_ - zero_idx - 1)
                * sizeof(int));
            --num_nonzero_doc_topic_idx_;
        }

        // 3. directly modify the local cache and updates
        topic_summary_[old_topic] -= 1;
        word_topic_count[word_start + old_topic] -= 1;
        word_topic_updates[topic_summary_start + old_topic] -= 1;
        word_topic_updates[word_start + old_topic] -= 1;

        // 4. Compute the mass in q bucket by iterating through the sparse word-topic
        // vector.
        q_sum_ = 0.;
        // Go through the list of word topic row with word id  equal to it.Word()
        // Update accordingly
        num_nonzero_q_terms_ = 0;
        for (int topic=0; topic<K_; ++topic) {
            int count = word_topic_count[word_start + topic];
            // skip zero count terms
            if (count > 0) {
                nonzero_q_terms_[num_nonzero_q_terms_] = (topic == old_topic) ?
                  (q_coeff_[topic] * (count - 1)) : (q_coeff_[topic] * count);
                nonzero_q_terms_topic_[num_nonzero_q_terms_] = topic;
                q_sum_ += nonzero_q_terms_[num_nonzero_q_terms_];
                ++num_nonzero_q_terms_;
            }
        }

        // Sample the topic for token 'it' using the aux variables.
        int new_topic = sample();
        assert(new_topic<K_);

        // 5. Add this token with new topic back to corpus using following steps:
        //
        // 5.1 Update s, r and q bucket.
        denom = topic_summary_[new_topic] + beta_sum_;
        s_sum_ -= (alpha_ * beta_) / denom;
        s_sum_ += (alpha_ * beta_) / (denom + 1);
        r_sum_ -= (theta_[new_topic] * beta_) / denom;
        r_sum_ += ((theta_[new_topic] + 1) * beta_) / (denom + 1);
        q_coeff_[new_topic] =
          (alpha_ + theta_[new_topic] + 1) / (denom + 1);


        // 5.2 theta_. If new_topic token count goes to 1, we need to add it
        // to the nonzero_doc_topic_idx_, etc.
        ++theta_[new_topic];   
        if (theta_[new_topic] == 1) {
            int* insert_idx =
              std::lower_bound(nonzero_doc_topic_idx_,
                  nonzero_doc_topic_idx_ + num_nonzero_doc_topic_idx_, new_topic);
            memmove(insert_idx + 1, insert_idx, (nonzero_doc_topic_idx_ +
                  num_nonzero_doc_topic_idx_ - insert_idx) * sizeof(int));
            *insert_idx = new_topic;
            ++num_nonzero_doc_topic_idx_;
        }

        // directly modify cache and updates
        topic_summary_[new_topic] += 1;
        word_topic_count[word_start + new_topic] += 1;
        word_topic_updates[topic_summary_start + new_topic] += 1;
        word_topic_updates[word_start + new_topic] += 1;

        // Finally, update the topic assignment z in doc
        if (old_topic != new_topic) {
            it.SetTopic(new_topic);
        }
    } // end of doc itration
}


// ====================== Private Functions ===================

void DocSampler::compute_doc_topic_vector(LDADoc& doc) {
    // Zero out theta_
    std::fill(theta_.begin(), theta_.end(), 0);
    for (LDADoc::Iterator it(&doc); !it.IsEnd(); it.Next()) {
        ++theta_[it.Topic()];
    }
}

 
void DocSampler::compute_aux_variables() {
    // zero out
    s_sum_ = 0.;
    r_sum_ = 0.;
    num_nonzero_doc_topic_idx_ = 0;
    double alpha_beta = alpha_ * beta_;

    for (int k = 0; k < K_; ++k) {
        double denom = topic_summary_[k] + beta_sum_;
        q_coeff_[k] = (alpha_ + theta_[k]) / denom;
        s_sum_ += alpha_beta / denom;

        if (theta_[k] != 0) {
            r_sum_ += (theta_[k] * beta_) / denom;
            // Populate nonzero_doc_topic_idx_.
            nonzero_doc_topic_idx_[num_nonzero_doc_topic_idx_++] = k;
        }
    }
}

int DocSampler::sample() {
    // Shooting a dart on [q_sum_ | r_sum_ | s_sum_] interval.
    double total_mass = q_sum_ + r_sum_ + s_sum_;
    double sample = uniform_zero_one_dist_(*rng_engine_) * total_mass;

    if (sample < q_sum_) {
        // The dart falls in [q_sum_ interval], which consists of [large_q_term |
        // ... | small_q_term]. ~90% should fall in the q bucket.
        for (int i = 0; i < num_nonzero_q_terms_; ++i) {
            sample -= nonzero_q_terms_[i];
            if (sample <= 0.) {
              return nonzero_q_terms_topic_[i];
            }
        }
        // Overflow.
        //LOG(INFO) << "sample = " << sample << " has overflowed in q bucket.";
        return nonzero_q_terms_topic_[num_nonzero_q_terms_ - 1];
    } else {
        sample -= q_sum_;
        if (sample < r_sum_) {
            // r bucket.
            sample /= beta_;  // save multiplying beta_ later on.
            for (int i = 0; i < num_nonzero_doc_topic_idx_; ++i) {
                int nonzero_topic = nonzero_doc_topic_idx_[i];
                sample -= theta_[nonzero_topic]
                  / (topic_summary_[nonzero_topic] + beta_sum_);
                if (sample <= 0.) {
                    return nonzero_topic;
                }
            }
            //LOG(INFO) << "sample = " << sample << " has overflowed in r bucket.";
            return nonzero_doc_topic_idx_[num_nonzero_doc_topic_idx_ - 1];
        } else {
            // s bucket.
            sample -= r_sum_;
            sample /= alpha_ * beta_;

            // There's no sparsity here to exploit. Just go through each term.
            for (int k = 0; k < K_; ++k) {
                sample -= 1. / (topic_summary_[k] + beta_sum_);
                if (sample <= 0.) {
                    return k;
                }
            }
            //LOG(INFO) << "sample = " << sample << " has overflowed in s bucket.";
            return (K_ - 1);
        }
    }
    return -1;
}

}  // namespace husky
