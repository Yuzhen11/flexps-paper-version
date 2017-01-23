/*
 * Parameters
 *
 * input
 * type: double
 * info: hdfs input file path
 *
 * num_features
 * type: int
 * info: number of features
 *
 * example:
 * input=hdfs:///1155014536/a9
 * num_feature:123
 *
 */

#include "datastore/datastore.hpp"
#include "worker/engine.hpp"

#include "lib/load_data.hpp"

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", "input",
                                          "hdfs_namenode", "hdfs_namenode_port", "num_features"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    datastore::DataStore<LabeledPointHObj<double, double, true>> data_store(
        Context::get_worker_info().get_num_local_workers());

    auto task = TaskFactory::Get().CreateTask<HuskyTask>(1, 4);  // 1 epoch, 4 workers
    engine.AddTask(std::move(task), [&data_store](const Info& info) {
        // load
        int num_features = std::stoi(Context::get_param("num_features"));
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, local_id);
    });
    engine.Submit();

    auto task2 = TaskFactory::Get().CreateTask<HuskyTask>(1, 4);
    engine.AddTask(std::move(task2), [&data_store](const Info& info) {
        // read from datastore
        int counter = 0;
        auto& local_data = data_store.Pull(info.get_local_id());
        for (auto& data : local_data) {
            counter += 1;
        }
        husky::LOG_I << "record num: " << std::to_string(counter);
    });
    engine.Submit();
    engine.Exit();
}
