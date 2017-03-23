#include <chrono>
#include <vector>

#include "worker/engine.hpp"
#include "ml/ml.hpp"
#include "lib/sample_reader.hpp"
#include "lib/task_utils.hpp"
#include "lib/app_config.hpp"
#include "husky/io/input/binary_inputformat.hpp"

using namespace husky;

int main(int argc, char** argv) {
    // Set config
    config::InitContext(argc, argv, {"parse", "is_binary"});
    auto config = config::SetAppConfigWithContext();
    if (Context::get_worker_info().get_process_id() == 0)
        config:: ShowConfig(config);
    auto hint = config::ExtractHint(config);

    auto& engine = Engine::Get();
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    auto task1 = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>();
    if (config.num_train_workers == 1 && config.kType == husky::constants::kSingle && config.kLoadHdfsType == "load_hdfs_locally") {
        task1.set_worker_num({1});
        task1.set_worker_num_type({"threads_traverse_cluster"});
    } else {
        task1.set_type(husky::Task::Type::MLTaskType);
    }
    task1.set_dimensions(config.num_params);
    task1.set_total_epoch(config.train_epoch);  // set epoch number
    task1.set_num_workers(config.num_train_workers);
    // Create KVStore and Set hint
    int kv1 = create_kvstore_and_set_hint(hint, task1, config.num_params);
    assert(kv1 != -1);

    bool is_binary = Context::get_param("is_binary") == "on" ? true : false;
    bool parse = Context::get_param("parse") == "on"? true : false;
    husky::LOG_I << "is_binary: " << is_binary << " parse: " << parse;
    engine.AddTask(task1, [is_binary, parse, config](const Info& info) {
        if (is_binary == false) {
            io::LineInputFormatML infmt(config.num_train_workers, info.get_task_id());
            infmt.set_input(Context::get_param("input"));
            typename io::LineInputFormat::RecordT record;

            int parse_time = 0;
            int read_count = 0;
            while(infmt.next(record)) {
                // parse
                if (parse) {
                    auto start_time = std::chrono::steady_clock::now();
                    boost::char_separator<char> sep(" \t");
                    boost::tokenizer<boost::char_separator<char>> tok(record, sep);
                    bool is_y = true;
                    for (auto& w : tok) {
                        if (!is_y) {
                            boost::char_separator<char> sep2(":");
                            boost::tokenizer<boost::char_separator<char>> tok2(w, sep2);
                            auto it = tok2.begin();
                            int idx = std::stoi(*it++) - 1;  // feature index from 0 to num_fea - 1
                            double val = std::stod(*it++);
                        } else {
                            is_y = false;
                        }
                    }
                    auto end_time = std::chrono::steady_clock::now();
                    parse_time += std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();
                }
                read_count += 1;
                if (read_count != 0 && read_count%10000 == 0)
                    husky::LOG_I << "read_count: " << read_count;
            }
            if (parse)
                husky::LOG_I << "parse time: " << parse_time/1000 << " ms";
            
            husky::LOG_I << "read_count: " << read_count;
        } else {
            husky::io::BinaryInputFormat infmt(Context::get_param("input"), "");
            typename husky::io::BinaryInputFormat::RecordT record;  // BinaryInputFormatRecord
            int read_count = 0;
            while (infmt.next(record)) {
                husky::base::BinStream& bin = husky::io::BinaryInputFormat::recast(record);
                if (parse) {
                    float y;
                    std::vector<std::pair<int, float>> v;
                    while (bin.size()) {
                        bin >> y >> v;
                        // husky::LOG_I << y;
                        // for (auto p : v)
                        //     husky::LOG_I << p.first << " " << p.second;
                        read_count += 1;
                        if (read_count != 0 && read_count%10000 == 0)
                            husky::LOG_I << "read_count: " << read_count;
                    }
                }
            }
            husky::LOG_I << "read_count: " << read_count;
        }
    });

    // Submit Task
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count();
    husky::LOG_I << YELLOW("Elapsed time: "+std::to_string(train_time) + " ms");

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
