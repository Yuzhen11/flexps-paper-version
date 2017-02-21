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

    const int threads_per_worker = 1;
    // const int MAGIC = 3;
    const int MAGIC = 480189;
    const int num_latent_factor = 10;
    const int chunk_size = num_latent_factor;
    const float lambda = 0.01;
    const int num_iters = 10;
    const bool do_test = true;
    const int local_read_count = 1000;

    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
    // ChunkSize: num_latent_factor
    // MaxKey: chunk_size*num_nodes
    // TODO, need to partition the data nicely
    kvstore::RangeManager::Get().SetMaxKeyAndChunkSize(kv1, MAGIC*2*chunk_size, chunk_size);

    // All the process should have this task running
    auto task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>();
    task.set_worker_num({threads_per_worker});
    task.set_worker_num_type({"threads_per_worker"});
    engine.AddTask(task, [&data_store1, kv1, lambda, num_iters](const Info& info) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
        auto& range_manager = kvstore::RangeManager::Get();

        std::vector<husky::base::BinStream> send_buffer(info.get_worker_info().get_largest_tid()+1);
        // load
        int local_tid_ptr = 0;
        auto parse_func = [&local_tid_ptr, &info, &send_buffer, &range_manager, kv1](boost::string_ref& chunk) {
            // parse
            boost::char_separator<char> sep(" \t");
            boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
            auto it = tok.begin();
            int user = std::stoi(*it);
            it++;
            int item = std::stoi(*it) + MAGIC;
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
            dst = tids[local_tid_ptr++];
            local_tid_ptr %= tids.size();  // TODO based on the assumption that each process has the same number of threads
            send_buffer[dst] << user << item << rating;

            server_id = range_manager.GetServerFromChunk(kv1, item);
            pid = server_id % num_processes;
            tids = info.get_worker_info().get_tids_by_pid(pid);
            assert(tids.size() > 0);
            dst = tids[local_tid_ptr++];
            local_tid_ptr %= tids.size();
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
            if (count == local_read_count)
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
        std::unordered_map<int, Node> local_nodes;
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
        for (int iter = 0; iter < num_iters; ++ iter) {
            float local_rmse = 0;  // for do_test
            int num_local_edges = 0;  // for do_test

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
                }
                if (do_test) {  // if do test, also need the parameters
                    keys.insert(kv.first);
                }
            }
            if (iter > 1) {
                std::vector<size_t> chunk_ids{keys.begin(), keys.end()};
                std::vector<std::vector<float>> params(chunk_ids.size());
                std::vector<std::vector<float>*> p_params(chunk_ids.size());
                for (size_t i = 0; i < chunk_ids.size(); ++ i)
                    p_params[i] = &params[i];
                int ts = kvworker->PullChunks(kv1, chunk_ids, p_params, false);
                kvworker->Wait(kv1, ts);
                // print
                // for (size_t i = 0; i < chunk_ids.size(); ++ i) {
                //     husky::LOG_I << "keys: " << chunk_ids[i];
                //     for (auto elem : params[i]) {
                //         std::cout << elem << " ";
                //     }
                //     std::cout << std::endl;
                // }
                for (size_t i = 0; i < chunk_ids.size(); ++ i) {
                    other_factors.insert({chunk_ids[i], Eigen::Map<Eigen::VectorXf>(&params[i][0], num_latent_factor)});
                }
            } else {  // if iter == 0 or 1, random generate parameter
                for (auto key : keys) {
                    Eigen::VectorXf v;
                    v.resize(num_latent_factor);
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
                    Eigen::VectorXf sum_vec = Eigen::VectorXf::Zero(num_latent_factor);
                    for (auto p : kv.second.nbs) {  // all the neighbors
                        if (sum_mat.size() == 0) {
                            sum_mat.resize(num_latent_factor, num_latent_factor);
                            sum_mat.triangularView<Eigen::Upper>() = other_factors[p.first] * other_factors[p.first].transpose();
                        } else {
                            sum_mat.triangularView<Eigen::Upper>() += other_factors[p.first] * other_factors[p.first].transpose();
                        }
                        sum_vec += other_factors[p.first]*p.second;

                        // predict
                        if (do_test) {
                            float pred = other_factors[kv.first].dot(other_factors[p.first]);
                            if (pred > 5.) pred = 5.;
                            if (pred < 1.) pred = 1.;
                            // husky::LOG_I << BLUE("edges: " + std::to_string(kv.first)+" "+std::to_string(p.first));
                            // husky::LOG_I << BLUE("predict: " + std::to_string(pred) + " truth: " + std::to_string(p.second));
                            local_rmse += (pred - p.second)*(pred - p.second);
                            num_local_edges += 1;
                        }
                    }
                    float reg = lambda * kv.second.nbs.size();  // TODO
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
            // print
            // for (int i = 0; i < send_chunk_ids.size(); ++ i) {
            //     husky::LOG_I << "sending: " << send_chunk_ids[i];
            //     for (auto elem : send_params[i]) {
            //         std::cout << elem << " ";
            //     }
            //     std::cout << std::endl;
            // }
            int ts = kvworker->PushChunks(kv1, send_chunk_ids, p_send_params, false);
            kvworker->Wait(kv1, ts);

            // 8. Barrier
            mailbox->send_complete(0, iter+1, 
                    info.get_worker_info().get_local_tids(), info.get_worker_info().get_pids());
            while (mailbox->poll(0, iter+1)) {
                mailbox->recv(0, iter+1);
            }
            
            // 9. Test error
            if (do_test) {
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

