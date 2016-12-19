#include "datastore/datastore.hpp"
#include "worker/engine.hpp"
#include "husky/io/input/line_inputformat.hpp"

using namespace husky;


int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", "input", "hdfs_namenode", "hdfs_namenode_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Create DataStore
    datastore::DataStore<std::string> data_store1(Context::get_worker_info().get_num_local_workers());

    auto task = TaskFactory::Get().create_task(Task::Type::HuskyTaskType, 1, 1);
    engine.AddTask(std::move(task), [&data_store1](const Info& info) {
        // load
        auto parse_func = [](boost::string_ref& chunk) {
            if (chunk.size() == 0)
                return;
            husky::base::log_msg(chunk.to_string());
        };
        husky::io::LineInputFormat infmt;
        infmt.set_input(husky::Context::get_param("input"));

        // loading
        typename io::LineInputFormat::RecordT record;
        bool success = false;
        while (true) {
            success = infmt.next(record);
            if (success == false)
                break;
            parse_func(io::LineInputFormat::recast(record));
        }

    });
    engine.Submit();
    engine.Exit();
}
