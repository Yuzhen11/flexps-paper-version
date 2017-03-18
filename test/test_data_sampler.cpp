#include "datastore/datastore.hpp"
#include "datastore/datastore_utils.hpp"
#include "lib/load_data.hpp"
#include "worker/engine.hpp"
#include "husky/io/input/line_inputformat.hpp"
#include "core/color.hpp"

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", "input", "num_features", "hdfs_namenode", "hdfs_namenode_port"});
    if (!rt)
        return 1;

    int num_features = std::stoi(Context::get_param("num_features"));
    auto& engine = Engine::Get();

    // Create the DataStore
    datastore::DataStore<LabeledPointHObj<float, float, true>> data_store(Context::get_worker_info().get_num_local_workers());

    auto task = TaskFactory::Get().CreateTask<HuskyTask>(1, 1); // 1 epoch, 1 workers
    engine.AddTask(std::move(task), [&data_store, &num_features](const Info & info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, local_id);
    });

    engine.Submit();

    auto task1 = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>();
    task1.set_worker_num({3, 1, 1});
    task1.set_worker_num_type({"threads_per_worker", "threads_per_cluster", "threads_per_cluster"});
    engine.AddTask(std::move(task1), [&data_store](const Info & info) {
        // create a DataStoreWrapper
        datastore::DataStoreWrapper<LabeledPointHObj<float, float, true>> data_store_wrapper(data_store);
        if (data_store_wrapper.get_data_size() == 0) {
            return;  // return if there is no data
        }
        // cast task
        std::vector<int> worker_num = static_cast<const ConfigurableWorkersTask*>(info.get_task())->get_worker_num();
        std::vector<int> tids = info.get_worker_info().get_local_tids();
        // find the pos of local_id
        int pos;
        for (int k = 0; k < tids.size(); k++) {
            if (tids[k] == info.get_local_id()) {
                pos = k;
                break;
            }
        }

        int current_epoch = info.get_current_epoch();

        // Create a DataLoadBalance for SGD
        datastore::DataLoadBalance<LabeledPointHObj<float, float, true>> data_load_balance(data_store, worker_num.size(), pos);

        int sum = 0;
        while (data_load_balance.has_next()) {
            // get next data
            auto& data = data_load_balance.next();
            sum++;
        }

        husky::LOG_I << RED("The SUM is " + std::to_string(sum));
    });

    engine.Submit();
    engine.Exit();
}
