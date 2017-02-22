#include "datastore/datastore.hpp"
#include "husky/io/input/line_inputformat.hpp"
#include "worker/engine.hpp"

#include <boost/tokenizer.hpp>

#include <Eigen/Dense>
#include <chrono>

using namespace husky;
/*
 *
 * CustomerID: [1,480189]
 * MovieID: [1,17770]
 *
 * 100,480,507 ratings
 *
 */

struct Node {
    int id;
    std::vector<std::pair<int, float>> nbs;
};

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", "input",
                                          "hdfs_namenode", "hdfs_namenode_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Create DataStore
    datastore::DataStore<std::string> data_store1(Context::get_worker_info().get_num_local_workers());
    // Start kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context(), 2);

    const int kStartFromOne = 1;  // 1 for netflix, 0 for toy
    const int kNumUsers = 480189;
    const int kNumItems = 17770;

    // const int kStartFromOne = 0;  // 1 for netflix, 0 for toy
    // const int kNumUsers = 3;
    // const int kNumItems = 3;

    const int kNumNodes = kNumUsers + kNumItems;
    const int MAGIC = kNumUsers + kStartFromOne;

    const int kThreadsPerWorker = 4;
    const int kNumLatentFactor = 10;
    const int kChunkSize = kNumLatentFactor;
    const float kLambda = 0.01;
    const int kNumIters = 10;
    const bool kDoTest = true;
    const int kLocalReadCount = 1000;

    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
    // ChunkSize: kNumLatentFactor
    // MaxKey: kChunkSize*num_nodes
    // TODO, need to partition the data nicely
    // kvstore::RangeManager::Get().SetMaxKeyAndChunkSize(kv1, MAGIC*2*kChunkSize, kChunkSize);

    // server partition setup
    std::vector<kvstore::pslite::Range> server_key_ranges;
    std::vector<kvstore::pslite::Range> server_chunk_ranges;
    int num_servers = kvstore::RangeManager::Get().GetNumServers();
    husky::constants::Key max_key = (kNumNodes + kStartFromOne) * kChunkSize;
    const int kChunkNum = kNumNodes + kStartFromOne;
    // 1. Partition users to 0 - num_servers/2
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
    // 2. Partition item to num_server/2 - num_servers
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
    // 3. Set server_key_ranges
    int range_counter = 0;
    for (auto range : server_chunk_ranges) {
        if (Context::get_process_id() == 0)
            husky::LOG_I << range_counter << " ranges: " << range.begin() << " " << range.end();
        server_key_ranges.push_back(kvstore::pslite::Range(range.begin()*kChunkSize, range.end()*kChunkSize));
        range_counter += 1;
    }
    kvstore::RangeManager::Get().CustomizeRanges(kv1, max_key, kChunkSize, kChunkNum,
            server_key_ranges, server_chunk_ranges);

    // All the process should have this task running
    auto task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>();
    task.set_worker_num({kThreadsPerWorker});
    task.set_worker_num_type({"threads_per_worker"});
    engine.AddTask(task, [&data_store1, kv1, kLambda, kNumIters, kNumUsers](const Info& info) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
        auto& range_manager = kvstore::RangeManager::Get();

        std::vector<husky::base::BinStream> send_buffer(info.get_worker_info().get_largest_tid()+1);
        // load
        auto parse_func = [&info, &send_buffer, &range_manager, kv1, kNumUsers](boost::string_ref& chunk) {
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
            tids = info.get_worker_info().get_tids_by_pid(pid);
            assert(tids.size() > 0);
            dst = tids[0];  // TODO: BAD!!!! all source node should go to the same place, not balance at all
            send_buffer[dst] << user << item << rating;

            server_id = range_manager.GetServerFromChunk(kv1, item);
            pid = server_id % num_processes;
            tids = info.get_worker_info().get_tids_by_pid(pid);
            assert(tids.size() > 0);
            dst = tids[0];
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

        // 4. Main loop
        for (int iter = 0; iter < kNumIters; ++ iter) {
            float local_rmse = 0;  // for kDoTest
            int num_local_edges = 0;  // for kDoTest

            auto start_time = std::chrono::steady_clock::now();
            std::map<size_t, Eigen::VectorXf> other_factors;  // map from chunk_ids to params
            // 5. Pull from kvstore
            std::set<size_t> keys;
            for (auto& kv : local_nodes) {
                if ((iter%2 == 0 && kv.first < MAGIC) || 
                    (iter%2 == 1 && kv.first >= MAGIC)) {
                    for (auto& p : kv.second.nbs) {
                        keys.insert(p.first);
                    }
                    if (kDoTest) {  // if do test, also need the parameters
                        keys.insert(kv.first);
                    }
                }
            }
            if (iter > 0) {
                std::vector<size_t> chunk_ids{keys.begin(), keys.end()};
                std::vector<std::vector<float>> params(chunk_ids.size());
                std::vector<std::vector<float>*> p_params(chunk_ids.size());
                for (size_t i = 0; i < chunk_ids.size(); ++ i)
                    p_params[i] = &params[i];
                // Pull from kvstore
                int ts = kvworker->PullChunks(kv1, chunk_ids, p_params, false);
                kvworker->Wait(kv1, ts);
                for (size_t i = 0; i < chunk_ids.size(); ++ i) {
                    other_factors.insert({chunk_ids[i], Eigen::Map<Eigen::VectorXf>(&params[i][0], kNumLatentFactor)});
                }
            } else {  // if iter == 0, random generate parameter
                for (auto key : keys) {
                    Eigen::VectorXf v;
                    v.resize(kNumLatentFactor);
                    v.setRandom();
                    other_factors.insert({key, std::move(v)});
                }
            }

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
            
            // 9. Test error
            if (kDoTest) {
                // if (info.get_cluster_id() == 0)
                if (num_local_edges != 0)
                    husky::LOG_I << BLUE("cluster id: " + std::to_string(info.get_cluster_id()) + 
                            "; nums of local edges: " + std::to_string(num_local_edges) + 
                            "; local rmse: "+std::to_string(local_rmse));
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

