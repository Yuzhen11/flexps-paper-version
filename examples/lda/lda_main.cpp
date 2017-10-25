#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>
#include <set>
#include <map>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>

#include "boost/tokenizer.hpp"
#include "boost/lexical_cast.hpp"

#include "datastore/datastore.hpp"
#include "husky/io/input/line_inputformat.hpp"
#include "kvstore/kvstore.hpp" 
#include "worker/engine.hpp"

#include "examples/lda/lda_doc.hpp"
#include "examples/lda/doc_sampler.hpp"
#include "examples/lda/lda_stat.hpp"

#include "ml/ml.hpp"

/* * Parameters:
 * * input: The file containing the whole corpus.
 *
 * alpha: Dirichlet prior on document-topic vectors.
 *
 * beta: Dirchlet prior on word-topic vectors.
 *
 * num_topics: The number of topics to be extracted from the corpus.
 *
 * num_iterations: The program will go through all the docs in every iteration.
 *
 * max_voc_id: The max id of the vocalbulary.
 *
 * Example:
 *
 * #LDA parameters
 *
 * input=hdfs:///yidi/pre_processed_nytimes
 * alpha=0.1
 * beta=0.1
 * num_topics=100 
 * num_iterations=10
 * max_voc_id=102600
 *
 *
 * http://www.pdl.cmu.edu/PDL-FTP/BigLearning/CMU-PDL-15-105.pdf, page 11
 * 
 * PS varables:
 * topic_summary: Dimention K
 *                Record the number of all words for each topic across the corpus
 *                Serve as the denominator inside sampler
 *                Is refreshed every iteration
 * 
 * word_topic_: Dimention V * K V is the vocalbulary size
 *                   Record the number of each word belong to each topic across the corpus
 * 
 * For every iteration:
 *     PS: get topic_summary
 *     For every document doc_i:
 *         Compute theta_i from doc_i
 *         For each word w_ij:
 *             PS: Get the word_topic_table word_topic_w_ij indexed by w_ij
 *             set k1 as previous z_ij
 *             Sample k2 as new z_ij using sampler(theta_i, word_topic_w_ij, alpha, beta)
 *             If k1 != k2
 *                 PS: update word_topic_table[w_ij][k1] -= 1
 *                 PS: update word_topic_table[w_ij][k2] += 1
 *                 PS: update topic_summary[k1] -= 1
 *                 PS: update topic_summary[k2] += 1
 *                 local update theta_i
 */
using namespace husky;

void load_data(datastore::DataStore<LDADoc>& corpus, int num_topics, const Info& info) {
    auto local_id = info.get_local_id();
    auto parse_func1 = [&corpus, &info, local_id, num_topics](std::string& chunk) {
        if (chunk.size() == 0)
            return;
        husky::LDADoc new_doc;
        // parse doc_id pair1 pair2 ...
        boost::char_separator<char> sep(" \n");
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
        boost::tokenizer<boost::char_separator<char>>::iterator iter = tok.begin();
        // the first is doc id
        int doc_id = boost::lexical_cast<int>(*iter);
        ++iter;

        // parse w1:f1 w2:f2 w3:f3 ...
        for (; iter != tok.end(); ++iter) {
            std::string word_fre_pair = boost::lexical_cast<std::string>(*iter);
            boost::char_separator<char> sep(":");    
            boost::tokenizer<boost::char_separator<char>> tok2(word_fre_pair, sep);
            boost::tokenizer<boost::char_separator<char>>::iterator iter1, iter2 = tok2.begin();
            iter1 = iter2;
            ++iter2;
            int word = boost::lexical_cast<int>(*iter1);
            int frequency = boost::lexical_cast<int>(*iter2);

            assert(word >= 1);
            // -1 for word id starts from 0
            new_doc.append_word(word - 1, frequency);
        }
        // randomly initilize the topic for each word
        new_doc.random_init_topics(num_topics);
        corpus.Push(local_id, std::move(new_doc));
    };

    int doc_count = 0;
    std::vector<husky::base::BinStream> send_buffer(info.get_num_workers());
    auto parse_func = [&send_buffer, &info, &doc_count](boost::string_ref& chunk) {
        if (chunk.size() == 0) 
            return;
        std::string line(chunk.data(), chunk.size());
        // evenly assign docs to all threads
        int dst = doc_count % info.get_num_workers();
        send_buffer[dst] << line;
        doc_count++; // every doc occupies a line
    };

    husky::io::LineInputFormat infmt;
    infmt.set_input(husky::Context::get_param("input"));
    typename io::LineInputFormat::RecordT record;
    bool success = false;
    while (true) {
        success = infmt.next(record);
        if (success == false)
            break;
        parse_func(io::LineInputFormat::recast(record));
    }

    auto* mailbox = Context::get_mailbox(info.get_local_id());
    for (int i = 0; i<send_buffer.size(); i++) {
        int dst = info.get_tid(i);
        if (send_buffer[i].size() == 0)
            continue;
        mailbox->send(dst, 2, 0, send_buffer[i]); // params: dst, channel, progress, bin
    }
    mailbox->send_complete(2, 0, 
            info.get_worker_info().get_local_tids(), info.get_worker_info().get_pids()); // params: channel, progress, sender_tids, recv_tids

    while (mailbox->poll(2, 0)) {
        auto bin = mailbox->recv(2, 0);
        std::string line;
        while (bin.size() != 0) {
            bin >> line;
            parse_func1(line);
        }
    }
}

void init_tables(std::vector<husky::LDADoc>& corpus,
    std::unique_ptr<ml::mlworker::GenericMLWorker<int>>& mlworker, int num_topics, int max_voc_id) {
    // 0. Create key val map 
    std::map<int, int> word_topic_count; 
    int sum = 0;
    if (!corpus.empty()) {
        int topic_summary_start =  max_voc_id * num_topics; 
        for (auto& doc : corpus) {
            for (LDADoc::Iterator it(&doc); !it.IsEnd(); it.Next()) {
                int word = it.Word();
                int topic = it.Topic();
                // update word topic
                sum+=1;
                word_topic_count[word * num_topics + topic] += 1;
                word_topic_count[topic_summary_start + topic] += 1;
            }
        }
    }
    LOG_I<<"sum is:"<<sum;

    //1. Create keys. They are already sorted
    std::vector<husky::constants::Key> lda_keys;
    lda_keys.reserve(word_topic_count.size());
    for (auto& key_val : word_topic_count) {
        lda_keys.push_back(key_val.first);
    }
    std::vector<int> lda_updates(lda_keys.size(), 0);
    mlworker->Pull(lda_keys, &lda_updates);

    // 2. Update update buffer 
    int i = 0;
    for (auto& key_val : word_topic_count) {
        lda_updates[i] = key_val.second;
        i++;
    }
    // 3. Push updates
    mlworker->Push(lda_keys, lda_updates);
}

void batch_training_by_chunk(std::vector<husky::LDADoc>& corpus, int lda_stat_table,
        std::unique_ptr<ml::mlworker::GenericMLWorker<int>>& mlworker, int num_topics, int max_voc_id, int num_iterations,
        float alpha, float beta, const Info& info ) {
    // 1. Find local vocabs and only Push and Pull those
    // 2. Bosen parameters synchronization:
    //    1. every iteration(called work unit) go through the corpus num_iters_per_work_unit times.
    //    2. every work unit will call Clock() num_clocks_per_work_unit times.
    //
    //    Our current synchronization:
    //    Same as setting num_iters_per_work_unit = num_clocks_per_work_unit.

    /*  difine the constants used int this function*/ 
    int need_compute_llh = std::stoi(Context::get_param("compute_llh"));
    int num_batches = std::stoi(Context::get_param("num_batches"));
    int max_vocs_each_pull = std::stoi(Context::get_param("max_vocs_each_pull"));
    std::string result_write_path = Context::get_param("result_write_path");
    int num_workers = info.get_num_workers();

    int num_vocs_per_thread = max_voc_id/num_workers;
    int vocab_id_start =  info.get_cluster_id() * num_vocs_per_thread;
    int vocab_id_end = (info.get_cluster_id() + 1 == num_workers) ? max_voc_id : vocab_id_start + num_vocs_per_thread;

    std::chrono::duration<double> pull_time;
    std::chrono::duration<double> push_time;
    std::chrono::duration<double> sample_time;
    std::chrono::duration<double> train_time;

    std::chrono::duration<double> llh_push_time;
    std::chrono::duration<double> llh_pull_time;
    std::chrono::duration<double> mailbox_time;
    std::chrono::duration<double> llh_time;

    int lda_llh_idx = 0;
    int pull_time_idx = 1;
    int push_time_idx = 2;
    int sample_time_idx = 3;
    int train_time_idx = 4;

    int llh_pull_time_idx  = 5;
    int llh_push_time_idx = 6;
    int mailbox_time_idx = 7;
    int llh_time_idx = 8;

    std::vector<husky::constants::Key> stat_keys(9);
    std::iota(stat_keys.begin(), stat_keys.end(), 0);
    std::vector<float> llh_and_time_updates(9, 0.0);
    std::vector<float> llh_and_time(9, 0.0);

    auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());

    std::vector<std::vector<int>> word_topic_count;
    std::vector<std::vector<int>> word_topic_updates;
    std::map<int, int> find_start_idx; // map the word to the start index of in batch_voc_keys and word_topic_count

    for (int i=1; i<=num_iterations; i++) {
        auto start_time = std::chrono::steady_clock::now();

        //  Divide the corpus into batches to do batch update
        // [j * num_docs_per_batch, j * num_docs_per_batch + num_docs_per_batch) for first num_batches-1 
        // [(num_batches - 1) * num_docs_per_batch, corpus.size()) for last batch
        assert(num_batches > 0);
        int num_docs_per_batch = corpus.size() / num_batches;
        std::vector<int> batch_end(num_batches);
        std::vector<int> batch_begin(num_batches);
        for (int j = 0; j < num_batches; j ++) {
            batch_begin[j] = j * num_docs_per_batch; 
            batch_end[j]  = (j + 1 == num_batches) ? corpus.size() : batch_begin[j] +  num_docs_per_batch;
        }

        int num_keys_pulled = 0;
        for (int j = 0; j<num_batches; j++) {
            //-----------------------------------
            // 1. make keys for local vocs needed for sampling
            std::set<int> batch_voc_set;
            for (int k = batch_begin[j]; k < batch_end[j]; k++) {
                for (LDADoc::Iterator it(&(corpus[k])); !it.IsEnd(); it.Next()) {
                    // batch_voc_set will be sorted in ascending order
                    batch_voc_set.insert(it.Word());
                }
            }

            // --------------------------------------------
            // Further reduce the number of keys for each batch
            // LightLDA. It is like coordinate descent.
            auto iter = batch_voc_set.begin();
            while (iter != batch_voc_set.end()) {
                std::vector<husky::constants::Key> batch_voc_keys;
                batch_voc_keys.reserve((batch_voc_set.size()+1));
                find_start_idx.clear();

                //-------------------------------------------------------
                // Make keys for word_topic_table and topic_summary row.
                // Each word corresponds to one chunk.
                for (int count = 0; count < max_vocs_each_pull && iter != batch_voc_set.end(); count ++) {
                    // Put all topics count for this word in batch_voc_keys[count * num_topics, (count + 1) * num_topics)
                    batch_voc_keys.push_back(*iter);
                    iter++;
                }
                // ----------------------------------------------------------------
                // (optional) add those keys needed for computing llh.
                if (need_compute_llh) {
                    // reserve more space
                    batch_voc_keys.reserve((batch_voc_set.size() + num_vocs_per_thread + 1));
                    for (int m = vocab_id_start; m < vocab_id_end; m++) {
                        batch_voc_keys.push_back(m);
                    }
                    // remove duplicates
                    std::sort(batch_voc_keys.begin(), batch_voc_keys.end());
                    batch_voc_keys.erase(std::unique(batch_voc_keys.begin(), batch_voc_keys.end()), batch_voc_keys.end());
                }

                // calculate the key map
                for (int count = 0; count < batch_voc_keys.size(); count++) {
                    find_start_idx[batch_voc_keys[count]] = count;
                }
                batch_voc_keys.push_back(max_voc_id);
                find_start_idx[max_voc_id] = batch_voc_keys.size() - 1;
                num_keys_pulled = batch_voc_keys.size() * num_topics;

                // -----------------------------------------
                // clear previous count and updates
                word_topic_count = std::move(std::vector<std::vector<int>>(batch_voc_keys.size(), std::vector<int>(num_topics, 0)));
                word_topic_updates = std::move(std::vector<std::vector<int>>(batch_voc_keys.size(), std::vector<int>(num_topics, 0)));
                {
                    // Pull chunks from kvstore
                    std::vector<std::vector<int>*> word_topic_count_ptrs(batch_voc_keys.size());
                    for ( int i=0; i<word_topic_count.size(); i++) {
                        word_topic_count_ptrs[i] = &(word_topic_count[i]);
                    }
                    auto start_pull_time = std::chrono::steady_clock::now();
                    mlworker->PullChunks(batch_voc_keys, word_topic_count_ptrs);
                    auto end_pull_time = std::chrono::steady_clock::now();
                    pull_time = end_pull_time - start_pull_time ;
                    //LOG_I<<"iter:" << i << " thread id:" <<info.get_global_id()<< " time taken:" <<pull_time.count()<<" Key size:"<<batch_voc_keys.size();
                }

                ////--------------------------------------------------------
                //// Doc sampler is responsible for
                //// 1. Compute the updates stored it in word_topic_updates;
                //// 2. Sample new topics for each word in corpus
                auto start_sample_time = std::chrono::steady_clock::now();
                DocSampler doc_sampler(num_topics, max_voc_id, alpha, beta);
                for (int k = batch_begin[j]; k < batch_end[j]; k++) {
                    doc_sampler.sample_one_doc(corpus[k], word_topic_count, word_topic_updates, find_start_idx);
                }
                auto end_sample_time = std::chrono::steady_clock::now();
    
                sample_time = end_sample_time - start_sample_time;
                {
                    // Push the updates to kvstore.
                    std::vector<std::vector<int>*> word_topic_updates_ptrs(batch_voc_keys.size());
                    for ( int i=0; i<word_topic_updates.size(); i++) {
                        word_topic_updates_ptrs[i] = &(word_topic_updates[i]);
                    }
                    auto start_push_time = std::chrono::steady_clock::now();
                    mlworker->PushChunks(batch_voc_keys, word_topic_updates_ptrs);
                    auto end_push_time = std::chrono::steady_clock::now();
                    push_time =  end_push_time - start_push_time;
                }

            }
        } // end of batch iter
        auto end_train_time = std::chrono::steady_clock::now();
        train_time = end_train_time - start_time;

        //LOG_I<<"iter:"<<i<<" client:"<<info.get_cluster_id()<< " training time:"<<train_time.count();

        if (need_compute_llh) {
            std::vector<int> topic_summary(num_topics);
            int topic_summary_start = find_start_idx[max_voc_id];
            int sum = 0;
            for (int k = 0; k < num_topics; k++) {
                topic_summary[k] = word_topic_count[topic_summary_start][k];
                sum += topic_summary[k];
            }
            if (info.get_cluster_id() == 0) {
                LOG_I<<" sum is:"<<sum;
            }

            LDAStats lda_stats(topic_summary, num_topics, max_voc_id, alpha, beta);
            
            double sum_llh = 0;
            // -------------------------------------------------------------
            // 1. Word llh. Only computed in the last iteration of last epoch
            sum_llh += lda_stats.ComputeWordLLH(vocab_id_start, vocab_id_end, word_topic_count, find_start_idx); 
            double word_llh = sum_llh;

            // ---------------------------------------------------------------
            // 2. Doc llh. Each thread computes by going through local corpus.
            // Computed in the last iteration of every epoch
            for (auto& doc : corpus) {
                sum_llh += lda_stats.ComputeOneDocLLH(doc);
            }
            double doc_llh = sum_llh - word_llh;

            // 3. Word summary llh.
            // But is computed in every epoch and every iteration for PS
            if (info.get_cluster_id() == 0 ) {
                sum_llh += lda_stats.ComputeWordLLHSummary();
            }
            double summary_llh = sum_llh - doc_llh - word_llh;

            auto mailbox_start_time = std::chrono::steady_clock::now();

            auto* mailbox = Context::get_mailbox(info.get_local_id());
            husky::base::BinStream bin;
            bin << sum_llh;
            int progress_id = i + num_iterations * info.get_current_epoch();
            mailbox->send(info.get_tid(0), 1, progress_id, bin);
            mailbox->send_complete(1, progress_id, 
                    info.get_worker_info().get_local_tids(), info.get_worker_info().get_pids());

            if (info.get_cluster_id() == 0) {
                double agg_llh = 0;
                while (mailbox->poll(1, progress_id)) {
                    auto bin = mailbox->recv(1, progress_id);
                    double tmp = 0;
                    bin >> tmp;
                    if (!std::isnan(tmp)) {
                        agg_llh += tmp;
                    }
                }
                LOG_I<<"local word llh:"<<word_llh<<" local doc_llh:"<<doc_llh<<" summary_llh:"<<summary_llh<<" agg_llh:"<<agg_llh;
                llh_and_time_updates[lda_llh_idx] = agg_llh;
            }
            auto mailbox_end_time = std::chrono::steady_clock::now();
            mailbox_time = mailbox_end_time - mailbox_start_time;
        } // end of compute llh
        auto end_iter_time = std::chrono::steady_clock::now();
        

        if (info.get_cluster_id() == 0) {
            llh_time = end_iter_time - end_train_time;
            LOG_I<<num_keys_pulled<<"keys. Pull:"<<pull_time.count()<<"s Push:"<<push_time.count()<<"s Sampling:"<<sample_time.count()<<" Train(P+P+S):"<<train_time.count()<<"s " <<"llh_update: "<<llh_and_time_updates[lda_llh_idx];
            LOG_I<<"mail_box time:"<<mailbox_time.count()<<"s "<<"llh_total_time:"<<llh_time.count()<<"s";
            llh_and_time_updates[train_time_idx] = train_time.count();
            llh_and_time_updates[pull_time_idx] = pull_time.count();
            llh_and_time_updates[push_time_idx] = push_time.count();
            llh_and_time_updates[sample_time_idx] = sample_time.count();
            llh_and_time_updates[llh_pull_time_idx] = llh_pull_time.count();
            llh_and_time_updates[llh_push_time_idx] = llh_push_time.count();
            llh_and_time_updates[mailbox_time_idx] = mailbox_time.count();
            llh_and_time_updates[llh_time_idx] = llh_time.count();

            std::string update_msg = "llh and time updates\n";
            for (auto a : llh_and_time_updates) {
                update_msg += std::to_string(a) + " ";
            }
            LOG_I<<update_msg;
            int ts = kvworker->Push(lda_stat_table, stat_keys, llh_and_time_updates, true);
            kvworker->Wait(lda_stat_table, ts);

            std::fill(llh_and_time.begin(), llh_and_time.end(), 0);
            ts = kvworker->Pull(lda_stat_table, stat_keys, &llh_and_time, true);
            kvworker->Wait(lda_stat_table, ts);

            std::ofstream ofs;
            ofs.open(result_write_path, std::ofstream::out | std::ofstream::app);
            if (!ofs.is_open()) {
                LOG_I<<"Error~ cannot open file";
            }
            ofs << i <<"   ";
            ofs << llh_and_time[train_time_idx] <<" "<< llh_and_time[lda_llh_idx]<<" ";
            ofs << std::setw(4) << llh_and_time[mailbox_time_idx]<<"s "<<llh_and_time[llh_time_idx]<<"s ";
            ofs << std::setw(4) << llh_and_time[lda_llh_idx] << "s\n";
            ofs.close();

            update_msg = "epoch:" + std::to_string(info.get_current_epoch()) + "\nPush:\n";
            for (auto& a : llh_and_time) {
                a *= -1.0;
                update_msg += std::to_string(a) + " ";
            }
            LOG_I<<update_msg;

            // clear the llh and start next iteration
            ts = kvworker->Push(lda_stat_table, stat_keys, llh_and_time, true);
            kvworker->Wait(lda_stat_table, ts);
        }
    }  // end of iteration
}

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", "input",
                                          "hdfs_namenode", "hdfs_namenode_port", "compute_llh", "result_write_path",
                                          "alpha", "beta", "num_topics", "num_iterations", "num_load_workers", "num_train_workers", "max_voc_id", "staleness", 
                                          "num_batches", "max_vocs_each_pull", "consistency"}); 
    int num_topics = std::stoi(Context::get_param("num_topics"));
    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    int num_train_workers = std::stoi(Context::get_param("num_train_workers"));
    int max_voc_id = std::stoi(Context::get_param("max_voc_id"));
    int num_epochs = 1;
    int num_iterations= std::stoi(Context::get_param("num_iterations"));
    float alpha = std::stof(Context::get_param("alpha"));
    float beta = std::stof(Context::get_param("beta"));
    int staleness = std::stoi(Context::get_param("staleness")); 
    std::string consistency =  Context::get_param("consistency");

    if (!rt)
        return 1;

    // engine must be created before kvstore
    auto& engine = Engine::Get();
    // TODO bug in seting more than one server per process
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(), Context::get_zmq_context(), 4);
    //
    //  topic_summary_table: A K dimentional vector
    //             Store the number of words assigned to topic k accross the whole corpus
    //             
    //  word_topic_table: Stored in the form of a long vector
    //            Store the number of word v assigned to topic k 
    //            

    /* Loadtask */
    datastore::DataStore<LDADoc> corpus(Context::get_worker_info().get_num_local_workers());
    auto load_task = TaskFactory::Get().CreateTask<Task>(1, num_load_workers); // 1 epoch
    auto load_task_lambda = [&corpus, num_topics](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(corpus, num_topics, info);
    };
    engine.AddTask(load_task, load_task_lambda); 

    int lda_stat_table = kvstore::KVStore::Get().CreateKVStore<float>("default_add_vector", -1, -1, 9, 9); // dimension is 9 chunck size is also 9

    /* LDA Task */
    int dim = (max_voc_id+1) * num_topics;
    int lda_table = kvstore::KVStore::Get().CreateKVStore<int>("ssp_add_vector", num_train_workers, staleness, dim, num_topics); // chunk size is set to num_topics
    auto lda_task = TaskFactory::Get().CreateTask<Task>(num_epochs, num_train_workers);
    TableInfo lda_table_info {
        lda_table, dim,
        husky::ModeType::PS, 
        husky::Consistency::SSP, 
        husky::WorkerType::PSWorker, 
        husky::ParamType::None,
        staleness
    };
    
    engine.AddTask(lda_task, [&corpus, num_epochs,  lda_stat_table, lda_table, lda_table_info, num_topics, max_voc_id, num_iterations, alpha, beta](const Info& info) {
        int local_id = info.get_local_id();
        auto& local_corpus = corpus.get_local_data(local_id);
        auto mlworker = ml::CreateMLWorker<int>(info, lda_table_info);
        LOG_I<<info.get_cluster_id()<<" There are:"<<local_corpus.size()<<" docs locally"<<std::endl;

        //--------------------------------------------------
        // 0. Initialize two tables according to randomly assigned topics
        auto start_epoch_timer = std::chrono::steady_clock::now();
        init_tables(local_corpus, mlworker, num_topics, max_voc_id);

        // -----------------------------------
        // 1. statistic header only write once
        if (info.get_cluster_id() == 0) {
            std::ofstream ofs;
            ofs.open(Context::get_param("result_write_path"), std::ofstream::out | std::ofstream::app);
            ofs << Context::get_param("input") <<" num_topics:"<< Context::get_param("num_topics") <<" num_trian_workers:"<< Context::get_param("num_train_workers");
            ofs <<" staleness:" << Context::get_param("staleness");
            ofs << "\nnum_epochs:"<<num_epochs<<"\n";
            ofs << " iter | pull_t | push_t | samplet | P+S+S | mailboxt | llh_time | llh\n";
            ofs.close();
        }
        // 2. -----------------Main logic ---------------
        batch_training_by_chunk(local_corpus, lda_stat_table, mlworker, num_topics, max_voc_id, num_iterations, alpha, beta, info);

        //------------------------------------------------
        // 3.(For multiple epoch task) calculate the statistics
        auto end_epoch_timer = std::chrono::steady_clock::now();
        std::chrono::duration<double> one_epoch_time = end_epoch_timer - start_epoch_timer;
    });

    engine.Submit();
    engine.Exit();
    // Stop the kvstore, should stop before mailbox is down
    kvstore::KVStore::Get().Stop();
}
