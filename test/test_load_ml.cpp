
#include "datastore/datastore.hpp"
#include "io/input/line_inputformat_ml.hpp"
#include "worker/engine.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", "input",
                                          "hdfs_namenode", "hdfs_namenode_port", "kLoadHdfsType"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Create DataStore
    datastore::DataStore<std::string> data_store1(Context::get_worker_info().get_num_local_workers());

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

        husky::LOG_I << RED(" task0 read: "
            + std::to_string(count)
            + " records in total.");
    });

    auto task1 = TaskFactory::Get().CreateTask<HuskyTask>(1, 6);
    engine.AddTask(task1, [&data_store1, &task1](const Info& info) {
        // load
        auto parse_func = [](boost::string_ref& chunk) {
            if (chunk.size() == 0)
                return;
            // husky::LOG_I << chunk.to_string();
        };
        io::LineInputFormatML infmt(6, task1.get_id());
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
        
        husky::LOG_I << RED(" task0 read: "
            + std::to_string(count)
            + " records in total.");
    });

    engine.Submit();
    engine.Exit();
}
