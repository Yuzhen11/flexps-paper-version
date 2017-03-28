#include "lib/app_config.hpp"
#include "datastore/datastore.hpp"
#include "io/input/line_inputformat_ml.hpp"
#include "io/input/binary_inputformat_ml.hpp"
#include "worker/engine.hpp"

using namespace husky;

int main(int argc, char** argv) {
    auto config = config::SetAppConfigWithContext();

    auto& engine = Engine::Get();
    // Create DataStore
    datastore::DataStore<std::string> data_store1(Context::get_worker_info().get_num_local_workers());

    // hdfs_load_block task
    /*
    auto task = TaskFactory::Get().CreateTask<HuskyTask>(1, 4);
    engine.AddTask(task, [&data_store1, &task](const Info& info) {
        // load
        auto parse_func = [](boost::string_ref& chunk) {
            if (chunk.size() == 0)
                return;
            // husky::LOG_I << chunk.to_string();
        };
        io::LineInputFormatML infmt(4, task.get_id());
        infmt.set_input(husky::Context::get_param("input"));

        // loading
        typename io::LineInputFormat::RecordT record;
        bool success = false;
        int count = 0;
        while (true) {
            success = infmt.next(record);
            if (success == false)
                break;
            parse_func(io::LineInputFormat::recast(record));
            count++;
        }

        husky::LOG_I << RED("hdfs_load_block task0 read: "
            + std::to_string(count)
            + " records in total.");
    });
    */
    
    bool parse = true;
    // hdfs_load_binary task
    
    // test for single
    auto task1 = TaskFactory::Get().CreateTask<HuskyTask>(1, 3);
    
    // test for running task on pointed thread
    /*
    auto task1 = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>(1, 3);
    task1.set_worker_num({1});
    task1.set_worker_num_type({"threads_on_worker:14"});
    */
    engine.AddTask(task1, [&data_store1, &task1, parse, &config](const Info& info) {
        // load
        auto parse_func = [](husky::base::BinStream& bin) {
            if (bin.size() == 0)
                return;
            // float y;
            // std::vector<std::pair<int, float>> v;

            // husky::LOG_I << chunk.to_string();
        };
        io::BinaryInputFormatML infmt(Context::get_param("input"), config.num_train_workers, task1.get_id());

        // loading
        typename io::BinaryInputFormatML::RecordT record;

        int read_count = 0;
        while (infmt.next(record)) {
            husky::base::BinStream& bin = husky::io::BinaryInputFormatML::recast(record);
            if (parse) {
                float y;
                std::vector<std::pair<int, float>> v;
                while (bin.size()) {
                    bin >> y >> v;
                    husky::LOG_I << y;
                    for (auto p : v)
                         husky::LOG_I << p.first << " " << p.second;
                    read_count += 1;
                    // husky::LOG_I << base::deser<std::string>(bin);
                    if (read_count != 0 && read_count%10000 == 0)
                        husky::LOG_I << "read_count: " << read_count;
                }
            }
        }
        
        husky::LOG_I << RED("hdfs_load_binary task1 read: "
            + std::to_string(read_count)
            + " records in total.");
    });

    engine.Submit();
    engine.Exit();
}
