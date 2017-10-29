#include <stdlib.h> /* srand, rand */
#include <time.h>
#include "datastore/datastore.hpp"
#include "husky/io/input/line_inputformat.hpp"
#include "kvstore/kvstore.hpp"
#include "worker/engine.hpp"

#include <boost/tokenizer.hpp>

#include <Eigen/Dense>
#include <chrono>

#ifdef USE_PROFILER
#include <gperftools/profiler.h>
#endif

using namespace husky;

/*
 *  #### Test MF_SGD
 *  # input=hdfs:///yuzhen/als_toy.txt
 *  # num_users=3
 *  # num_items=3
 *  # start_from_one=0
 *
 *  # netflix
 *  # input=hdfs:///datasets/ml/netflix
 *  # num_users=480189
 *  # num_items=17770
 *  # start_from_one=1
 *
 *  # yahoomusic
 *  # input=hdfs:///datasets/ml/yahoomusic
 *  # num_users=1823179
 *  # num_items=136736
 *  # start_from_one=0
 *
 *  # num_workers_per_process=10
 *  # num_servers_per_process=10
 *  # num_iter=10
 *  # num_latent_factor=10
 */
struct Node {
    int user;
    int item;
    float value;
};  // Node is one entry of matrix: user represents rowID, item represents colID

int main(int argc, char** argv) {
    bool rt = init_with_args(
        argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", "input", "hdfs_namenode",
                     "hdfs_namenode_port", "num_users", "num_items", "start_from_one", "num_workers_per_process",
                     "num_servers_per_process", "num_iters", "num_latent_factors"});
    if (!rt)
        return 1;

    const int kStartFromOne = std::stoi(Context::get_param("start_from_one"));
    const int kNumUsers = std::stoi(Context::get_param("num_users"));
    const int kNumItems = std::stoi(Context::get_param("num_items"));
    const int kWorkersPerProcess = std::stoi(Context::get_param("num_workers_per_process"));
    const int kServersPerProcess = std::stoi(Context::get_param("num_servers_per_process"));  // 20 is the largest
    const int kNumIters = std::stoi(Context::get_param("num_iters"));
    const int kNumLatentFactor = std::stoi(Context::get_param("num_latent_factors"));

    const int kNumNodes = kNumUsers + kNumItems;
    const int MAGIC = kNumUsers + kStartFromOne;

    const int kChunkSize = kNumLatentFactor;
    const float kLambda = 0.01;
    const float kLearningRate = 0.05;  // You could change this paramter slightly, but not serverely
    const bool kDoTest = true;
    const int kLocalReadCount = -1;

    auto& engine = Engine::Get();
    // Create DataStore
    datastore::DataStore<std::string> data_store1(Context::get_worker_info().get_num_local_workers());
    // Start kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context(), kServersPerProcess);

    // 1. Server partition setup
    std::vector<kvstore::pslite::Range> server_key_ranges;
    std::vector<kvstore::pslite::Range> server_chunk_ranges;
    int num_servers = kvstore::RangeManager::Get().GetNumServers();
    husky::constants::Key max_key = (kNumNodes + kStartFromOne) * kChunkSize;
    const int kChunkNum = kNumNodes + kStartFromOne;
    // 1.1 Partition users to 0 - num_servers/2
    int num_half_servers = num_servers / 2;
    int chunk_num = (kNumUsers + kStartFromOne);
    int base = chunk_num / num_half_servers;
    int remain = chunk_num % num_half_servers;
    for (int i = 0; i < remain; ++i) {
        server_chunk_ranges.push_back(kvstore::pslite::Range(i * (base + 1), (i + 1) * (base + 1)));
    }
    int end = remain * (base + 1);
    for (int i = 0; i < num_half_servers - remain - 1; ++i) {
        server_chunk_ranges.push_back(kvstore::pslite::Range(end + i * base, end + (i + 1) * (base)));
    }
    server_chunk_ranges.push_back(kvstore::pslite::Range(end + (num_half_servers - remain - 1) * base, chunk_num));
    // 1.2. Partition item to num_server/2 - num_servers
    num_half_servers = num_servers - num_half_servers;
    chunk_num = (kNumItems + kStartFromOne);
    base = chunk_num / num_half_servers;
    remain = chunk_num % num_half_servers;
    int offset = kNumUsers + kStartFromOne;
    for (int i = 0; i < remain; ++i) {
        server_chunk_ranges.push_back(kvstore::pslite::Range(offset + i * (base + 1), offset + (i + 1) * (base + 1)));
    }
    end = remain * (base + 1);
    for (int i = 0; i < num_half_servers - remain - 1; ++i) {
        server_chunk_ranges.push_back(kvstore::pslite::Range(offset + end + i * base, offset + end + (i + 1) * (base)));
    }
    server_chunk_ranges.push_back(
        kvstore::pslite::Range(offset + end + (num_half_servers - remain - 1) * base, offset + chunk_num));
    // 1.3. Set server_key_ranges
    int range_counter = 0;
    for (auto range : server_chunk_ranges) {
        if (Context::get_process_id() == 0)
            husky::LOG_I << range_counter << " ranges: " << range.begin() << " " << range.end();
        server_key_ranges.push_back(kvstore::pslite::Range(range.begin() * kChunkSize, range.end() * kChunkSize));
        range_counter += 1;
    }
    // 1.4 Create kvstore
    int kv1 = kvstore::KVStore::Get().CreateKVStoreWithoutSetup();
    // 1.5 CustomizeRanges
    kvstore::RangeManager::Get().CustomizeRanges(kv1, max_key, kChunkSize, kChunkNum, server_key_ranges,
                                                 server_chunk_ranges);
    // 1.6 Setup kvstore
    kvstore::KVStore::Get().SetupKVStore<float>(kv1, "default_assign_vector", -1, -1);

    // All the process should have this task running
    auto task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>();
    task.set_worker_num({kWorkersPerProcess});
    task.set_worker_num_type({"threads_per_worker"});
    engine.AddTask(task, [&data_store1, kv1, kLambda, kLearningRate, kNumIters, kNumLatentFactor, kWorkersPerProcess,
                          kNumUsers, kChunkNum, MAGIC](const Info& info) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
        auto& range_manager = kvstore::RangeManager::Get();

        std::vector<husky::base::BinStream> send_buffer(info.get_worker_info().get_largest_tid() + 1);
        // load
        auto parse_func = [&info, &send_buffer, &range_manager, kv1, kNumUsers,
                           kWorkersPerProcess](boost::string_ref& chunk) {
            // parse
            boost::char_separator<char> sep(" \t");
            boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
            auto it = tok.begin();
            int user = std::stoi(*it);
            it++;
            int item = std::stoi(*it) + kNumUsers;
            it++;
            float rating = std::stof(*it);
            // husky::LOG_I << user << " " << item << " " << rating;

            // Push into send_buffer
            // Need to carefully handle the destination
            int server_id = -1;
            int pid = -1;
            int dst = -1;
            int num_processes = info.get_worker_info().get_num_processes();
            std::vector<int> tids;

            server_id = range_manager.GetServerFromChunk(kv1, user);
            pid = server_id % num_processes;
            tids = info.get_worker_info().get_tids_by_pid(pid);  // the result tids must be the same in each machine
            assert(tids.size() == kWorkersPerProcess);
            dst = tids[user % kWorkersPerProcess];  // first locate the process, then hash partition
            send_buffer[dst] << user << item << rating;
        };
        husky::io::LineInputFormat infmt;
        infmt.set_input(husky::Context::get_param("input"));

        auto start_time = std::chrono::steady_clock::now();
        // 1. Load the data and push into send_buffer
        typename io::LineInputFormat::RecordT record;
        bool success = false;
        int count = 0;
        while (true) {
            success = infmt.next(record);
            if (success == false)
                break;
            parse_func(io::LineInputFormat::recast(record));
            count += 1;
            if (count == kLocalReadCount)
                break;
        }

        // 2. Flush the send_buffer
        auto* mailbox = Context::get_mailbox(info.get_local_id());
        int start = info.get_global_id();
        for (int i = 0; i < send_buffer.size(); ++i) {
            int dst = (start + i) % send_buffer.size();
            if (send_buffer[dst].size() == 0)
                continue;
            mailbox->send(dst, 0, 0, send_buffer[dst]);
            send_buffer[dst].purge();
        }
        mailbox->send_complete(0, 0, info.get_worker_info().get_local_tids(), info.get_worker_info().get_pids());

        int worker_number = send_buffer.size();
        // 3. Receive and Store the data
        std::vector<Node> local_nodes[worker_number];  // use array to regulate the further operation
        while (mailbox->poll(0, 0)) {
            auto bin = mailbox->recv(0, 0);
            int from, to;
            float r;
            while (bin.size() != 0) {
                bin >> from >> to >> r;
                // husky::LOG_I << "recv: " << from << " " << to << " " << r;
                Node n = {from, to, r};
                local_nodes[to % worker_number].push_back(std::move(n));
            }
        }

        auto end_time = std::chrono::steady_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (info.get_cluster_id() == 0) {
            husky::LOG_I << CLAY("Load done. Load time is: " + std::to_string(total_time) + " ms");
        }
        // Prepare pull keys
        std::set<size_t> keys[send_buffer.size()];  // keys for different iteration or permutation of block
        std::vector<size_t> chunk_ids[worker_number];
        std::vector<std::vector<float>> params[worker_number];
        std::vector<std::vector<float>*> p_params[worker_number];
        for (size_t i = 0; i < worker_number; i++) {
            for (auto& kv : local_nodes[i]) {
                keys[i].insert(kv.user);
                keys[i].insert(kv.item);
            }
            chunk_ids[i] = {keys[i].begin(), keys[i].end()};
            params[i].resize(chunk_ids[i].size());
            p_params[i].resize(chunk_ids[i].size());
            for (size_t j = 0; j < chunk_ids[i].size(); ++j)
                p_params[i][j] = &params[i][j];
        }
        std::vector<Eigen::VectorXf> other_factors(kChunkNum);
        float initial_learning_rate = 0.005;
        float learning_rate_decaying = 0.99;
        // 4. Main loop
        for (int iter = 0; iter < kNumIters; ++iter) {
            float local_rmse = 0;                                      // for kDoTest
            int num_local_edges = 0;                                   // for kDoTest
            int idx = (iter + info.get_cluster_id()) % worker_number;  // Do the permutation
            auto start_time = std::chrono::steady_clock::now();
            // 5. Pull from kvstore
            if (iter > 0) {
                initial_learning_rate = initial_learning_rate * learning_rate_decaying;
                int ts = kvworker->PullChunks(kv1, chunk_ids[idx], p_params[idx], false);
                kvworker->Wait(kv1, ts);
                for (size_t i = 0; i < chunk_ids[idx].size(); ++i) {
                    // other_factors.insert({chunk_ids[idx][i], Eigen::Map<Eigen::VectorXf>(&params[idx][i][0],
                    // kNumLatentFactor)});
                    other_factors[chunk_ids[idx][i]] =
                        Eigen::Map<Eigen::VectorXf>(&params[idx][i][0], kNumLatentFactor);
                }

            } else {  // if iter == 0, random generate parameter
                for (size_t i = 0; i < worker_number; i++) {
                    for (auto key : keys[i]) {
                        Eigen::VectorXf v;
                        v.resize(kNumLatentFactor);
                        v.setRandom();
                        v = v + Eigen::VectorXf::Ones(kNumLatentFactor);
                        // other_factors.insert({key, std::move(v)});
                        other_factors[key] = std::move(v);
                    }
                }
            }
            auto prepare_end_time = std::chrono::steady_clock::now();
            auto prepare_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(prepare_end_time - start_time).count();
            if (info.get_cluster_id() == 0)
                husky::LOG_I << CLAY("[Iter] " + std::to_string(iter) + " prepare time: " +
                                     std::to_string(prepare_time) + " ms");

            // 6. Update
            std::vector<size_t> send_chunk_ids;
            std::vector<std::vector<float>> send_params;
            std::vector<std::vector<float>*> p_send_params;
            // iterate over the nodes of block and do SGD on it
            for (auto& kv : local_nodes[idx]) {
                Eigen::VectorXf user_gradient = Eigen::VectorXf::Zero(kNumLatentFactor);
                Eigen::VectorXf item_gradient = Eigen::VectorXf::Zero(kNumLatentFactor);

                float eij = kv.value - other_factors[kv.user].dot(other_factors[kv.item]);
                user_gradient = initial_learning_rate * eij * other_factors[kv.item];
                item_gradient = initial_learning_rate * eij * other_factors[kv.user];
                other_factors[kv.user] += user_gradient;
                other_factors[kv.item] += item_gradient;
                if (kDoTest) {
                    float pred = other_factors[kv.user].dot(other_factors[kv.item]);
                    if (pred > 5.)
                        pred = 5.;
                    if (pred < 1.)
                        pred = 1.;
                    // if (kv.first == 106859) {
                    //     std::cout << "id: " << kv.first << " : ";
                    //     for (int i = 0; i < kNumLatentFactor; ++ i) {
                    //         std::cout << other_factors[kv.first][i] << " ";
                    //     }
                    //     std::cout << std::endl;
                    //
                    //     std::cout << "id: " << p.first << " : ";
                    //     for (int i = 0; i < kNumLatentFactor; ++ i) {
                    //         std::cout << other_factors[p.first][i] << " ";
                    //     }
                    //     std::cout << std::endl;
                    //     husky::LOG_I << BLUE("edges: " + std::to_string(kv.first)+" "+std::to_string(p.first));
                    //     husky::LOG_I << BLUE("predict: " + std::to_string(pred) + " truth: " +
                    //     std::to_string(p.second));
                    // }
                    local_rmse += (pred - kv.value) * (pred - kv.value);
                    num_local_edges += 1;
                }

                // send_chunk_ids.push_back(kv.user);
                // send_params.push_back(std::vector<float>(user_gradient.data(),
                // user_gradient.data()+user_gradient.rows()*user_gradient.cols()));  // copy
                // send_chunk_ids.push_back(kv.item);
                // send_params.push_back(std::vector<float>(item_gradient.data(),
                // item_gradient.data()+item_gradient.rows()*item_gradient.cols()));
            }
            for (auto key : chunk_ids[idx])
                send_params.push_back(std::vector<float>(
                    other_factors[key].data(),
                    other_factors[key].data() + other_factors[key].rows() * other_factors[key].cols()));
            auto compute_end_time = std::chrono::steady_clock::now();
            auto compute_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(compute_end_time - prepare_end_time).count();
            if (info.get_cluster_id() == 0)
                husky::LOG_I << CLAY("[Iter] " + std::to_string(iter) + " compute time: " +
                                     std::to_string(compute_time) + " ms");

            p_send_params.reserve(send_chunk_ids.size());
            for (auto& vec : send_params) {
                p_send_params.push_back(&vec);
            }
            // 7. Push to kvstore
            int ts = kvworker->PushChunks(kv1, chunk_ids[idx], p_send_params, false);
            kvworker->Wait(kv1, ts);

            // 8. Barrier
            if (kDoTest) {  // aggregate global rmse
                int dst = info.get_tid(0);
                husky::base::BinStream bin;
                bin << local_rmse << num_local_edges;
                mailbox->send(dst, 0, iter + 1, bin);
            }
            mailbox->send_complete(0, iter + 1, info.get_worker_info().get_local_tids(),
                                   info.get_worker_info().get_pids());
            if (kDoTest) {
                float total_rmse = 0;
                int total_num_edges = 0;
                while (mailbox->poll(0, iter + 1)) {
                    auto bin = mailbox->recv(0, iter + 1);
                    if (info.get_cluster_id() == 0) {
                        float rmse_part;
                        int num_edges_part;
                        bin >> rmse_part >> num_edges_part;
                        total_rmse += rmse_part;
                        total_num_edges += num_edges_part;
                    }
                }
                if (info.get_cluster_id() == 0) {
                    husky::LOG_I << BLUE("global avg rmse: " + std::to_string(total_rmse / total_num_edges));
                }
            } else {
                while (mailbox->poll(0, iter + 1)) {
                    mailbox->recv(0, iter + 1);
                }
            }

            // 9. Test error
            if (kDoTest) {
                // if (info.get_cluster_id() == 0)
                if (num_local_edges != 0 && info.get_cluster_id() < 5) {
                    husky::LOG_I << BLUE("cluster id: " + std::to_string(info.get_cluster_id()) +
                                         "; nums of local edges: " + std::to_string(num_local_edges) +
                                         "; local rmse: " + std::to_string(local_rmse) + "; aveg rmse: " +
                                         std::to_string(local_rmse / num_local_edges));
                }

                auto end_time = std::chrono::steady_clock::now();
                auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
                if (info.get_cluster_id() == 0) {
                    husky::LOG_I << CLAY("[Iter] " + std::to_string(iter) + " time: " + std::to_string(total_time) +
                                         " ms");
                }
            }
        }
    });
    engine.Submit();
    engine.Exit();
    // Stop the kvstore, should stop before mailbox is down
    kvstore::KVStore::Get().Stop();
}
