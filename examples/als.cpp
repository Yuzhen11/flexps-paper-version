#include "datastore/datastore.hpp"
#include "husky/io/input/line_inputformat.hpp"
#include "worker/engine.hpp"
#include "kvstore/kvstore.hpp"

#include <boost/tokenizer.hpp>

#include <Eigen/Dense>
#include <chrono>

#ifdef USE_PROFILER
#include <gperftools/profiler.h>
#endif

using namespace husky;

/*
 *  #### Test ALS
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
    int id;
    std::vector<std::pair<int, float>> nbs;
};

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", "input",
                                          "hdfs_namenode", "hdfs_namenode_port",
                                          "num_users", "num_items", "start_from_one",
                                          "num_workers_per_process", "num_servers_per_process",
                                          "num_iters", "num_latent_factors"});
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
    int num_half_servers = num_servers/2;
    int chunk_num = (kNumUsers + kStartFromOne);
    int base = chunk_num / num_half_servers;
    int remain = chunk_num % num_half_servers;
    for (int i = 0; i < remain; ++ i) {
        server_chunk_ranges.push_back(kvstore::pslite::Range(i*(base+1), (i+1)*(base+1)));
    }
    int end = remain*(base+1);
    for (int i = 0; i < num_half_servers - remain - 1; ++ i) {
        server_chunk_ranges.push_back(kvstore::pslite::Range(end+i*base, end+(i+1)*(base)));
    }
    server_chunk_ranges.push_back(kvstore::pslite::Range(end+(num_half_servers-remain-1)*base, chunk_num));
    // 1.2. Partition item to num_server/2 - num_servers
    num_half_servers = num_servers - num_half_servers;
    chunk_num = (kNumItems + kStartFromOne);
    base = chunk_num / num_half_servers;
    remain = chunk_num % num_half_servers;
    int offset = kNumUsers + kStartFromOne;
    for (int i = 0; i < remain; ++ i) {
        server_chunk_ranges.push_back(kvstore::pslite::Range(offset+i*(base+1), offset+(i+1)*(base+1)));
    }
    end = remain*(base+1);
    for (int i = 0; i < num_half_servers - remain - 1; ++ i) {
        server_chunk_ranges.push_back(kvstore::pslite::Range(offset+end+i*base, offset+end+(i+1)*(base)));
    }
    server_chunk_ranges.push_back(kvstore::pslite::Range(offset+end+(num_half_servers-remain-1)*base, offset+chunk_num));
    // 1.3. Set server_key_ranges
    int range_counter = 0;
    for (auto range : server_chunk_ranges) {
        if (Context::get_process_id() == 0)
            husky::LOG_I << range_counter << " ranges: " << range.begin() << " " << range.end();
        server_key_ranges.push_back(kvstore::pslite::Range(range.begin()*kChunkSize, range.end()*kChunkSize));
        range_counter += 1;
    }
    // 1.4 Create kvstore
    int kv1 = kvstore::KVStore::Get().CreateKVStoreWithoutSetup();
    // 1.5 CustomizeRanges
    kvstore::RangeManager::Get().CustomizeRanges(kv1, max_key, kChunkSize, kChunkNum,
            server_key_ranges, server_chunk_ranges);
    // 1.6 Setup kvstore
    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kStorageType, husky::constants::kVectorStorage},
        {husky::constants::kUpdateType, husky::constants::kAssignUpdate}
    };
    kvstore::KVStore::Get().SetupKVStore<float>(kv1, hint);

    // All the process should have this task running
    auto task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>();
    task.set_worker_num({kWorkersPerProcess});
    task.set_worker_num_type({"threads_per_worker"});
    engine.AddTask(task, [&data_store1, kv1, kLambda, kNumIters, kNumLatentFactor, kWorkersPerProcess, kNumUsers, kChunkNum, MAGIC](const Info& info) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
        auto& range_manager = kvstore::RangeManager::Get();

        std::vector<husky::base::BinStream> send_buffer(info.get_worker_info().get_largest_tid()+1);
        // load
        auto parse_func = [&info, &send_buffer, &range_manager, kv1, kNumUsers, kWorkersPerProcess](boost::string_ref& chunk) {
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

            server_id = range_manager.GetServerFromChunk(kv1, item);
            pid = server_id % num_processes;
            tids = info.get_worker_info().get_tids_by_pid(pid);
            assert(tids.size() == kWorkersPerProcess);
            dst = tids[item % kWorkersPerProcess];
            send_buffer[dst] << item << user << rating;
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
        for (int i = 0; i < send_buffer.size(); ++ i) {
            int dst = (start + i) % send_buffer.size();
            if (send_buffer[dst].size() == 0)
                continue;
            mailbox->send(dst, 0, 0, send_buffer[dst]);
            send_buffer[dst].purge();
        }
        mailbox->send_complete(0, 0, 
                info.get_worker_info().get_local_tids(), info.get_worker_info().get_pids());

        // 3. Receive and Store the data
        std::map<int, Node> local_nodes;  // use tree map to maintain the order
        while (mailbox->poll(0,0)) {
            auto bin = mailbox->recv(0,0);
            int from, to; float r;
            while (bin.size() != 0) {
                bin >> from >> to >> r;
                // husky::LOG_I << "recv: " << from << " " << to << " " << r;
                local_nodes[from].nbs.push_back({to, r});
            }
        }

        auto end_time = std::chrono::steady_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count();
        if (info.get_cluster_id() == 0) {
            husky::LOG_I << CLAY("Load done. Load time is: " + std::to_string(total_time) + " ms");
        }

        // Prepare pull keys
        std::set<size_t> keys[2];  // keys for odd/even iter
        for (auto& kv : local_nodes) {
            if (kv.first < MAGIC) {
                for (auto& p : kv.second.nbs) {
                    keys[0].insert(p.first);
                }
            } else {
                for (auto& p : kv.second.nbs) {
                    keys[1].insert(p.first);
                }
            }
            if (kDoTest) {  // if do test, also need the parameters, will be very slow
                keys[0].insert(kv.first);
                keys[1].insert(kv.first);
            }
        }
        std::vector<size_t> chunk_ids[2];
        chunk_ids[0] = {keys[0].begin(), keys[0].end()};
        chunk_ids[1] = {keys[1].begin(), keys[1].end()};
        std::vector<std::vector<float>> params[2];
        params[0].resize(chunk_ids[0].size());
        params[1].resize(chunk_ids[1].size());
        std::vector<std::vector<float>*> p_params[2];
        p_params[0].resize(chunk_ids[0].size());
        p_params[1].resize(chunk_ids[1].size());
        for (size_t i = 0; i < chunk_ids[0].size(); ++ i)
            p_params[0][i] = &params[0][i];
        for (size_t i = 0; i < chunk_ids[1].size(); ++ i)
            p_params[1][i] = &params[1][i];

        std::vector<Eigen::VectorXf> other_factors(kChunkNum);
        // 4. Main loop
        for (int iter = 0; iter < kNumIters; ++ iter) {
#ifdef USE_PROFILER
            if (info.get_cluster_id() == 0) {
                ProfilerStart(("/data/opt/tmp/yuzhen/als2/train-"+std::to_string(iter)+".prof").c_str());
            }
#endif
            float local_rmse = 0;  // for kDoTest
            int num_local_edges = 0;  // for kDoTest

            auto start_time = std::chrono::steady_clock::now();
            // 5. Pull from kvstore
            if (iter > 0) {
                // Pull from kvstore
                int idx = iter % 2;
                int ts = kvworker->PullChunks(kv1, chunk_ids[idx], p_params[idx], false);
                kvworker->Wait(kv1, ts);
                for (size_t i = 0; i < chunk_ids[idx].size(); ++ i) {
                    // other_factors.insert({chunk_ids[idx][i], Eigen::Map<Eigen::VectorXf>(&params[idx][i][0], kNumLatentFactor)});
                    other_factors[chunk_ids[idx][i]] = Eigen::Map<Eigen::VectorXf>(&params[idx][i][0], kNumLatentFactor);
                }
            } else {  // if iter == 0, random generate parameter
                for (auto key : keys[0]) {
                    Eigen::VectorXf v;
                    v.resize(kNumLatentFactor);
                    v.setRandom();
                    // other_factors.insert({key, std::move(v)});
                    other_factors[key] = std::move(v);
                }
            }
            auto prepare_end_time = std::chrono::steady_clock::now();
            auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(prepare_end_time-start_time).count();
            if (info.get_cluster_id() == 0)
                husky::LOG_I << CLAY("[Iter] "+std::to_string(iter) + " prepare time: " + std::to_string(prepare_time) + " ms");

            // 6. Update
            std::vector<size_t> send_chunk_ids;
            std::vector<std::vector<float>> send_params;
            std::vector<std::vector<float>*> p_send_params;
            // iterate over the nodes
            for (auto& kv : local_nodes) {
                if ((iter % 2 == 0 && kv.first < MAGIC) ||
                    (iter % 2 == 1 && kv.first >= MAGIC)) {
                    Eigen::MatrixXf sum_mat;
                    Eigen::VectorXf sum_vec = Eigen::VectorXf::Zero(kNumLatentFactor);
                    for (auto p : kv.second.nbs) {  // all the neighbors
                        if (sum_mat.size() == 0) {
                            sum_mat.resize(kNumLatentFactor, kNumLatentFactor);
                            sum_mat.triangularView<Eigen::Upper>() = other_factors[p.first] * other_factors[p.first].transpose();
                        } else {
                            sum_mat.triangularView<Eigen::Upper>() += other_factors[p.first] * other_factors[p.first].transpose();
                        }
                        sum_vec += other_factors[p.first]*p.second;

                        // predict
                        if (kDoTest) {
                            float pred = other_factors[kv.first].dot(other_factors[p.first]);
                            if (pred > 5.) pred = 5.;
                            if (pred < 1.) pred = 1.;
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
                            //     husky::LOG_I << BLUE("predict: " + std::to_string(pred) + " truth: " + std::to_string(p.second));
                            // }
                            local_rmse += (pred - p.second)*(pred - p.second);
                            num_local_edges += 1;
                        }
                    }
                    float reg = kLambda * kv.second.nbs.size();  // TODO
                    for (int i = 0; i < sum_mat.rows(); ++ i)
                        sum_mat(i,i) += reg;
                    Eigen::VectorXf factor = sum_mat.selfadjointView<Eigen::Upper>().ldlt().solve(sum_vec);
                    send_chunk_ids.push_back(kv.first);
                    send_params.push_back(std::vector<float>(factor.data(), factor.data()+factor.rows()*factor.cols()));  // copy
                }
            }
            auto compute_end_time = std::chrono::steady_clock::now();
            auto compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(compute_end_time-prepare_end_time).count();
            if (info.get_cluster_id() == 0)
                husky::LOG_I << CLAY("[Iter] "+std::to_string(iter) + " compute time: " + std::to_string(compute_time) + " ms");

            p_send_params.reserve(send_chunk_ids.size());
            for (auto& vec : send_params) {
                p_send_params.push_back(&vec);
            }
            // 7. Push to kvstore
            int ts = kvworker->PushChunks(kv1, send_chunk_ids, p_send_params, false);
            kvworker->Wait(kv1, ts);

            // 8. Barrier
            mailbox->send_complete(0, iter+1, 
                    info.get_worker_info().get_local_tids(), info.get_worker_info().get_pids());
            while (mailbox->poll(0, iter+1)) {
                mailbox->recv(0, iter+1);
            }

#ifdef USE_PROFILER
            if (info.get_cluster_id() == 0) {
                ProfilerStop();
            }
#endif
            
            // 9. Test error
            if (kDoTest) {
                // if (info.get_cluster_id() == 0)
                if (num_local_edges != 0 && info.get_cluster_id() < 5)
                    husky::LOG_I << BLUE("cluster id: " + std::to_string(info.get_cluster_id()) + 
                            "; nums of local edges: " + std::to_string(num_local_edges) + 
                            "; local rmse: " + std::to_string(local_rmse) +
                            "; aveg rmse: " + std::to_string(local_rmse/num_local_edges));
            }

            auto end_time = std::chrono::steady_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count();
            if (info.get_cluster_id() == 0)
                husky::LOG_I << CLAY("[Iter] "+std::to_string(iter) + " time: " + std::to_string(total_time) + " ms");
        }
    });
    engine.Submit();
    engine.Exit();
    // Stop the kvstore, should stop before mailbox is down
    kvstore::KVStore::Get().Stop();
}

