#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>
#include <set>
#include <map>
#include <sstream>
#include <fstream>

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
    if (!corpus.empty()) {
        int topic_summary_start =  max_voc_id * num_topics; 
        for (auto& doc : corpus) {
            for (LDADoc::Iterator it(&doc); !it.IsEnd(); it.Next()) {
                int word = it.Word();
                int topic = it.Topic();
                // update word topic
                word_topic_count[word * num_topics + topic] += 1;
                word_topic_count[topic_summary_start + topic] += 1;
            }
        }
    }

    //1. Create keys. They are already sorted
    std::vector<husky::constants::Key> lda_keys;
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
    int num_epochs = std::stoi(Context::get_param("num_epochs"));
    int num_process = std::stoi(Context::get_param("num_process"));
    int num_batches = std::stoi(Context::get_param("num_batches"));
    int max_vocs_each_pull = std::stoi(Context::get_param("max_vocs_each_pull"));
    std::string result_write_path = Context::get_param("result_write_path");

    std::chrono::duration<double> pull_time;
    std::chrono::duration<double> push_time;
    std::chrono::duration<double> sample_time;
    std::chrono::duration<double> train_time;
    std::chrono::duration<double> llh_time;

    int lda_llh_idx = 0;
    int total_pull_time = 1;
    int total_push_time = 2;
    int total_sample_time = 3;
    int total_train_time = 4;

    std::vector<husky::constants::Key> stat_keys(5);
    std::iota(stat_keys.begin(), stat_keys.end(), 0);
    std::vector<float> llh_and_time_updates(5, 0.0);
    std::vector<float> llh_and_time(5, 0.0);

    auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
    bool is_last_epoch = ((info.get_current_epoch() + 1) % num_process == 0);

    for (int i=1; i<=num_iterations; i++) {
        auto start_time = std::chrono::steady_clock::now();

        //  Divide the corpus into batches to do batch update
        assert(num_batches > 0);
        int num_docs_per_batch = corpus.size() / num_batches;
        std::vector<int> batch_end(num_batches);
        std::vector<int> batch_begin(num_batches);
        // [j * num_docs_per_batch, j * num_docs_per_batch + num_docs_per_batch) for first num_batches-1 
        // [(num_batches - 1) * num_docs_per_batch, corpus.size()) for last batch
        for (int j = 0; j < num_batches; j ++) {
            batch_begin[j] = j * num_docs_per_batch; 
            batch_end[j]  = (j + 1 == num_batches) ? corpus.size() : batch_begin[j] +  num_docs_per_batch;
        }

        int num_keys_pulled = 0;
        for (int j = 0; j<num_batches; j++) {
            std::set<int> batch_voc_set;
            for (int k = batch_begin[j]; k < batch_end[j]; k++) {
                for (LDADoc::Iterator it(&(corpus[k])); !it.IsEnd(); it.Next()) {
                    // batch_voc_set will be sorted in ascending order
                    batch_voc_set.insert(it.Word());
                }
            }

            // Further reduce the number of keys for each batch
            // LightLDA. It is like coordinate descent.
            auto iter = batch_voc_set.begin();
            int partition_count = -1;
            std::vector<husky::constants::Key> batch_voc_keys;
            while (iter != batch_voc_set.end()) {
                partition_count++;
                batch_voc_keys.reserve((batch_voc_set.size()+1));
                std::map<int, int> find_start_idx; // map the word to the start index of in batch_voc_keys and word_topic_count
                int last_idx = 0;
                for (int count = 0; count < max_vocs_each_pull && iter != batch_voc_set.end(); count ++) {
                    // Put all topics count for this word in batch_voc_keys[count * num_topics, (count + 1) * num_topics)
                    find_start_idx[*iter]  = count;
                    batch_voc_keys.push_back(*iter);
                    iter++;
                    last_idx++;
                }
                num_keys_pulled = find_start_idx.size() * num_topics;

                // Handle the word_topic_summary row. Treat it as the last k items both in the kvstore and locally
                find_start_idx[max_voc_id] = last_idx;
                batch_voc_keys.push_back(max_voc_id);

                std::vector<std::vector<int>> word_topic_count(find_start_idx.size(), std::vector<int>(num_topics, 0));
                std::vector<std::vector<int>> word_topic_updates(find_start_idx.size(), std::vector<int>(num_topics, 0));

                {
                    std::vector<std::vector<int>*> word_topic_count_ptrs(batch_voc_keys.size());
                    for ( int i=0; i<word_topic_count.size(); i++) {
                        word_topic_count_ptrs[i] = &(word_topic_count[i]);
                    }
                    auto start_pull_time = std::chrono::steady_clock::now();
                    mlworker->PullChunks(batch_voc_keys, word_topic_count_ptrs);
                    auto end_pull_time = std::chrono::steady_clock::now();
                    pull_time = end_pull_time - start_pull_time ;
                }

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
                // 4. Push the updates to kvstore.
                {
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
        }

        auto end_train_time = std::chrono::steady_clock::now();
        train_time = end_train_time - start_time;

        if (need_compute_llh) {
            // 1. Each thread is responsible for computing part of whole word llh.
            int num_workers = info.get_num_workers();
            int num_vocs_per_thread = max_voc_id/num_workers;
            int vocab_id_start =  info.get_cluster_id() * num_vocs_per_thread;
            int vocab_id_end = (info.get_cluster_id() + 1 == num_workers) ? max_voc_id : vocab_id_start + num_vocs_per_thread;

            std::vector<husky::constants::Key> llh_keys;
            for (int m = vocab_id_start; m < vocab_id_end; m++) {
                for (int k = 0; k < num_topics; k++)
                    llh_keys.push_back(m * num_topics + k);
            }
            // for topic summary row 
            for (int k = 0; k < num_topics; k++)
                llh_keys.push_back(max_voc_id * num_topics + k);

            std::vector<int> llh_word_topic_count(llh_keys.size(), 0);
            mlworker->Pull(llh_keys, &llh_word_topic_count);
            for (int i = 0; i < llh_word_topic_count.size(); i++) {
                assert(llh_word_topic_count[i] >= 0);
            }


            std::vector<int> topic_summary(num_topics);
            int topic_summary_start = (vocab_id_end  - vocab_id_start) * num_topics;
            for (int k = 0; k < num_topics; k++)
                topic_summary[k] = llh_word_topic_count[topic_summary_start + k];

            LDAStats lda_stats(topic_summary, num_topics, max_voc_id, alpha, beta);
            
            double sum_llh = 0;
            // 1. Word llh. Only computed in the last iteration of last epoch
            if ((i == num_iterations && is_last_epoch) || (Context::get_param("kType") == "PS")) {
                sum_llh += lda_stats.ComputeWordLLH(vocab_id_start, vocab_id_end, llh_word_topic_count); 
            }
            double word_llh = sum_llh;

            // 2. Doc llh. Each thread computes by going through local corpus. Computed in the last iteration of every epoch
            if (i == num_iterations || Context::get_param("kType") == "PS") {
                for (auto& doc : corpus) {
                    sum_llh += lda_stats.ComputeOneDocLLH(doc);
                }
            }
            double doc_llh = sum_llh - word_llh;

            // 3. Word summary llh. Only one thread computes it at the end of all epochs for SPMT
            // But is computed in every epoch and every iteration for PS
            if ((info.get_cluster_id() == 0 && is_last_epoch && i == num_iterations) || (info.get_cluster_id() == 0 && Context::get_param("kType") == "PS")) {
                sum_llh += lda_stats.ComputeWordLLHSummary();
                LOG_I<<"Computed word_llh + local_doc_llh + local_word_llh is equal to "<<sum_llh;
            }

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
                    agg_llh += tmp;
                }
                // Only the last iteration sum all word llhs
                if (i == num_iterations || Context::get_param("kType") == "PS") {
                    LOG_I<<"local word llh:"<<word_llh<<" local doc_llh:"<<doc_llh<<" agg_llh:"<<agg_llh;
                    llh_and_time_updates[lda_llh_idx] = agg_llh;
                }
            }
            mlworker->Push(llh_keys, std::vector<int>(llh_keys.size(), 0));
        } // end of compute llh

        auto end_iter_time = std::chrono::steady_clock::now();
        if (info.get_cluster_id() == 0) {
            llh_time = end_iter_time - end_train_time;
            LOG_I<<num_keys_pulled<<"keys. Pull:"<<pull_time.count()<<"s Push:"<<push_time.count()<<"s Sampling:"<<sample_time.count()<<" Train(P+P+S):"<<train_time.count()<<"s llh:" << llh_time.count() <<"s " <<"llh_update: "<<llh_and_time_updates[lda_llh_idx];
            llh_and_time_updates[total_train_time] = train_time.count();
            llh_and_time_updates[total_pull_time] = pull_time.count();
            llh_and_time_updates[total_push_time] = push_time.count();
            llh_and_time_updates[total_sample_time] = sample_time.count();
            int ts = kvworker->Push(lda_stat_table, stat_keys, llh_and_time_updates, true);
            kvworker->Wait(lda_stat_table, ts);
        }

        // only the last iteration of last epoch, write to the file
        if ((info.get_cluster_id() == 0 && is_last_epoch && i == num_iterations) || (info.get_cluster_id() == 0 && Context::get_param("kType") == "PS")) {
            //LOG_I<<Context::get_param("result_write_path");
            std::ofstream ofs;
            ofs.open(result_write_path, std::ofstream::out | std::ofstream::app);
            if (!ofs.is_open()) {
                LOG_I<<"Error~ cannot open file";
            }
            int num_workers = info.get_num_workers();

            std::fill(llh_and_time.begin(), llh_and_time.end(), 0);
            int ts = kvworker->Pull(lda_stat_table, stat_keys, &llh_and_time, true);
            kvworker->Wait(lda_stat_table, ts);
            int iteration = (Context::get_param("kType") == "PS") ? i : (info.get_current_epoch() / num_process + 1);
            ofs << iteration <<"\t"<< llh_and_time[total_pull_time] <<"s\t"<< llh_and_time[total_push_time] <<"s\t" << llh_and_time[total_sample_time] << "s\t" << llh_and_time[total_train_time] << "s\t"<< llh_and_time[lda_llh_idx] <<"\n";

            LOG_I << "epoch:"<<info.get_current_epoch()<<" "<< llh_and_time[lda_llh_idx];

            for (auto& a : llh_and_time)
                a *= -1.0;
            // clear the llh and start next iteration
            ts = kvworker->Push(lda_stat_table, stat_keys, llh_and_time, true);
            kvworker->Wait(lda_stat_table, ts);
            ofs.close();
        }
    }  // end of iteration
}

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", "input",
                                          "hdfs_namenode", "hdfs_namenode_port", "compute_llh", "result_write_path", "num_epochs", "num_process", "kType",
                                          "alpha", "beta", "num_topics", "num_iterations", "num_load_workers", "num_train_workers", "max_voc_id", "staleness", 
                                          "num_batches", "max_vocs_each_pull", "consistency"}); 
    int num_topics = std::stoi(Context::get_param("num_topics"));
    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    int num_train_workers = std::stoi(Context::get_param("num_train_workers"));
    int max_voc_id = std::stoi(Context::get_param("max_voc_id"));
    int num_epochs = std::stoi(Context::get_param("num_epochs"));
    int num_iterations= std::stoi(Context::get_param("num_iterations"));
    float alpha = std::stof(Context::get_param("alpha"));
    float beta = std::stof(Context::get_param("beta"));
    std::string staleness = Context::get_param("staleness"); 
    std::string kType = Context::get_param("kType");
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

    // Create the DataStore
    datastore::DataStore<LDADoc> corpus(Context::get_worker_info().get_num_local_workers());
    // Loadtask
    auto load_task = TaskFactory::Get().CreateTask<HuskyTask>(1, num_load_workers);  // 1 epoch
    auto load_task_lambda = [&corpus, num_topics](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(corpus, num_topics, info);
    };
    load_task.set_num_workers(num_load_workers);
    engine.AddTask(load_task, load_task_lambda); 
    //engine.Submit(); cannot submit twice?

    std::map<std::string, std::string> stat_hint = 
    {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kStorageType, husky::constants::kVectorStorage},
        {husky::constants::kNumWorkers, "1"},
        {husky::constants::kConsistency, husky::constants::kASP}
    };
    int lda_stat_table = kvstore::KVStore::Get().CreateKVStore<float>(stat_hint, 5, 5); // chunk size is set to num_topics

    std::map<std::string, std::string> hint = 
    {
        //{husky::constants::kType, husky::constants::kPS},
        {husky::constants::kType, kType},
        //{husky::constants::kParamType, husky::constants::kChunkType},
        {husky::constants::kEnableDirectModelTransfer, "true"},

        
        //{husky::constants::kStaleness, staleness},
        
        {husky::constants::kStorageType, husky::constants::kVectorStorage},
        {husky::constants::kNumWorkers, std::to_string(num_train_workers)},

        //{husky::constants::kWorkerType, husky::constants::kPSWorker},
        
        //{husky::constants::kConsistency, husky::constants::kSSP}
        {husky::constants::kConsistency, consistency}
        //{husky::constants::kConsistency, husky::constants::kASP}
    };

    // set up kvstore
    int lda_table = kvstore::KVStore::Get().CreateKVStore<int>(hint, (max_voc_id+1) * num_topics, num_topics); // chunk size is set to num_topics

    auto lda_task = TaskFactory::Get().CreateTask<MLTask>(); // 1 epoch 10 threads
    lda_task.set_hint(hint);
    lda_task.set_kvstore(lda_table); 
    lda_task.set_dimensions((max_voc_id + 1) * num_topics);
    lda_task.set_num_workers(num_train_workers);
    lda_task.set_total_epoch(num_epochs);

    std::ofstream ofs;
    ofs.open(Context::get_param("result_write_path"), std::ofstream::out | std::ofstream::app);
    ofs << Context::get_param("input") <<" num_topics:"<< Context::get_param("num_topics") <<" num_trian_workers:"<< Context::get_param("num_train_workers");
    ofs << "\nnum_epochs:"<<num_epochs<<" num_process"<<Context::get_param("num_process")<<" kType:"<<Context::get_param("kType") << "\n";
    ofs << "iter\tpull_time\tpush_time\tsampling_time\ttrain_time(P+S+S)\tllh\n";
    ofs.close();

    engine.AddTask(lda_task, [&corpus, lda_stat_table, lda_table, num_topics, max_voc_id, num_iterations, alpha, beta](const Info& info) {
        int local_id = info.get_local_id();
        auto& local_corpus = corpus.get_local_data(local_id);
        auto mlworker = ml::CreateMLWorker<int>(info);
        // LOG_I<<info.get_cluster_id()<<" There are:"<<local_corpus.size()<<" docs locally"<<std::endl;

        auto start_epoch_timer = std::chrono::steady_clock::now();
        // 1. Initialize two tables according to randomly assigned topics
        int num_process = std::stoi(Context::get_param("num_process"));
        if (info.get_current_epoch() / num_process == 0 || Context::get_param("kType") == "PS") {
            init_tables(local_corpus, mlworker, num_topics, max_voc_id);
        }

        // 2. Main logic
        // LOG_I<<info.get_cluster_id()<<" start training ";
        batch_training_by_chunk(local_corpus, lda_stat_table, mlworker, num_topics, max_voc_id, num_iterations, alpha, beta, info);
        auto end_epoch_timer = std::chrono::steady_clock::now();

        std::chrono::duration<double> one_epoch_time = end_epoch_timer - start_epoch_timer;
        if (info.get_cluster_id() == 0) {
            std::ofstream ofs;
            ofs.open(Context::get_param("result_write_path"), std::ofstream::out | std::ofstream::app);
            ofs<<"epoch "<<info.get_current_epoch()<<" takes:"<<one_epoch_time.count()<<"s\n";
            LOG_I<<"one_epoch_time:"<< one_epoch_time.count();
            ofs.close();
        }
    });

    engine.Submit();
    engine.Exit();
    // Stop the kvstore, should stop before mailbox is down
    kvstore::KVStore::Get().Stop();
}
