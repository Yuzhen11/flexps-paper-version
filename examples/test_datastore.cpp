#include "datastore/datastore.hpp"
#include "worker/engine.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port"});
    if (!rt)
        return 1;

    Engine engine;
    datastore::DataStore<std::string> data_store1(Context::get_worker_info().get_num_local_workers());

    auto task = TaskFactory::Get().create_task(Task::Type::HuskyTaskType, 1, 4);
    engine.add_task(std::move(task), [&data_store1](const Info& info) {
        // load
        // write to datastore
        data_store1.Push(info.get_local_id(), "hello world from "+std::to_string(info.get_local_id()));
    });
    engine.submit();

    auto task2 = TaskFactory::Get().create_task(Task::Type::HuskyTaskType, 1, 4);
    engine.add_task(std::move(task2), [&data_store1](const Info& info) {
        // read from datastore
        auto& local_data = data_store1.Pull(info.get_local_id());
        for (auto& data : local_data) {
            husky::base::log_msg("data 1: "+data);
        }
    });
    engine.submit();
    engine.exit();
}
