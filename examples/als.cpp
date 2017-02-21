#include "datastore/datastore.hpp"
#include "husky/io/input/line_inputformat.hpp"
#include "worker/engine.hpp"

#include <boost/tokenizer.hpp>

#include <Eigen/Dense>

using namespace husky;

struct Node {
    int id;
    std::vector<std::pair<int, double>> nbs;
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

    const int threads_per_worker = 2;
    const int MAGIC = 3;

    // TODO
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
    kvstore::RangeManager::Get().SetMaxKeyAndChunkSize(kv1, MAGIC*2, 1);

    // All the process should have this task running
    auto task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>();
    task.set_worker_num({threads_per_worker});
    task.set_worker_num_type({"threads_per_worker"});
    engine.AddTask(task, [&data_store1, kv1](const Info& info) {
        std::vector<husky::base::BinStream> send_buffer(info.get_worker_info().get_largest_tid()+1);
        auto& range_manager = kvstore::RangeManager::Get();
        // load
        auto parse_func = [&info, &send_buffer, &range_manager, kv1](boost::string_ref& chunk) {
            // parse
            boost::char_separator<char> sep(" \t");
            boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
            auto it = tok.begin();
            int user = std::stoi(*it);
            it++;
            int item = std::stoi(*it) + MAGIC;
            it++;
            double rating = std::stof(*it);
            husky::LOG_I << user << " " << item << " " << rating;

            // Push into send_buffer
            // Need to carefully handle the destination
            int server_id = -1;
            int pid = -1;
            int dst = -1;
            int num_processes = info.get_worker_info().get_num_processes();
            std::vector<int> tids;

            server_id = range_manager.GetServer(kv1, user);
            pid = server_id % num_processes;
            tids = info.get_worker_info().get_tids_by_pid(pid);
            assert(tids.size() > 0);
            dst = tids[0];  // TODO now just select the first one
            send_buffer[dst] << user << item << rating;

            server_id = range_manager.GetServer(kv1, item);
            pid = server_id % num_processes;
            tids = info.get_worker_info().get_tids_by_pid(pid);
            assert(tids.size() > 0);
            dst = tids[0];
            send_buffer[dst] << item << user << rating;
        };
        husky::io::LineInputFormat infmt;
        infmt.set_input(husky::Context::get_param("input"));

        // 1. Load the data and push into send_buffer
        typename io::LineInputFormat::RecordT record;
        bool success = false;
        while (true) {
            success = infmt.next(record);
            if (success == false)
                break;
            parse_func(io::LineInputFormat::recast(record));
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
            int from, to; double r;
            while (bin.size() != 0) {
                bin >> from >> to >> r;
                husky::LOG_I << "recv: " << from << " " << to << " " << r;
                local_nodes[from].nbs.push_back({to, r});
            }
        }

    });
    engine.Submit();
    engine.Exit();
    // Stop the kvstore, should stop before mailbox is down
    kvstore::KVStore::Get().Stop();
}

