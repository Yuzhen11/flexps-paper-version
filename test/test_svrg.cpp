#include <vector>

#include "datastore/datastore.hpp"
#include "datastore/datastore_utils.hpp"
#include "worker/engine.hpp"
#include "ml/common/mlworker.hpp"
#include "kvstore/kvstore.hpp"
#include "core/color.hpp"

#include "lib/load_data.hpp"

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

using namespace husky;

void test_error(std::vector<float>& rets_w,
    datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store) {

    datastore::DataIterator<LabeledPointHObj<float, float, true>> data_iterator(data_store);
    
    std::string debug_kvstore_w;
    int flag = 1;
    int count = 0;
    float c_count = 0; /// correct count
    while (data_iterator.has_next()) {
        // get next data
        auto& data = data_iterator.next();

        count = count + 1;
        auto& x = data.x;
        float y = data.y;
        if (y < 0) y = 0;
        float pred_y = 0.0;

        for (auto field : x) {
            pred_y += rets_w[field.fea] * field.val;
            if (flag) { debug_kvstore_w += "__" + std::to_string(rets_w[field.fea]); }
        }
        pred_y = 1. / (1. + exp(-1 * pred_y));

        pred_y = (pred_y > 0.5) ? 1 : 0;
        if (int(pred_y) == int(y)) { c_count += 1;}
        flag = 0;
    }

    husky::LOG_I << "current w: " + debug_kvstore_w;
    husky::LOG_I << ":accuracy is " << std::to_string(c_count / count) 
        << " count is :" << std::to_string(count) << " c_count is:" << std::to_string(c_count);
}

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port",
                                          "hdfs_namenode", "hdfs_namenode_port",
                                          "input", "num_features", "alpha", "num_iters",
                                          "train_epoch"
                                         });

    int train_epoch = std::stoi(Context::get_param("train_epoch"));
    float alpha = std::stof(Context::get_param("alpha"));
    int num_iters = std::stoi(Context::get_param("num_iters"));
    int num_features = std::stoi(Context::get_param("num_features"));
    int num_params = num_features + 1; // +1 for intercept

    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // start the kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    // Create the DataStore
    datastore::DataStore<LabeledPointHObj<float, float, true>> data_store(Context::get_worker_info().get_num_local_workers());

    auto task = TaskFactory::Get().CreateTask<HuskyTask>(1, 1); // 1 epoch, 1 workers
    engine.AddTask(std::move(task), [&data_store, &num_features](const Info & info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, local_id);
        husky::LOG_I << RED("Finished Load Data!");
    });

    engine.Submit();

    auto task1 = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>(3, 6);
    task1.set_worker_num({3, 1, 1});
    task1.set_worker_num_type({"threads_per_worker", "threads_traverse_cluster", "threads_traverse_cluster"});
    int kv_w = kvstore::KVStore::Get().CreateKVStore<float>();
    int kv_u = kvstore::KVStore::Get().CreateKVStore<float>();
    engine.AddTask(std::move(task1), [&kv_w, &kv_u, &data_store, num_iters, alpha, num_params](const Info & info) {
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
        std::vector<husky::constants::Key> keys;
        for (int i = 0; i < num_params; i++) { keys.push_back(i); }
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
        // Create a DataLoadBalance for SGD
        datastore::DataLoadBalance<LabeledPointHObj<float, float, true>> data_load_balance(data_store, worker_num.size(), pos);
        data_load_balance.start_point();
        if (current_epoch % worker_num.size() == 0) {
            // do FGD
            // pull from kvstore_w
            std::vector<float> rets_w;
            kvworker->Wait(kv_w, kvworker->Pull(kv_w, keys, &rets_w));

            std::vector<float> delta(num_params);
            float sum = 0;
            while (data_load_balance.has_next()) {
                sum++;
                // get next data
                auto& data = data_load_balance.next();
                auto& x = data.x;
                float y = data.y;
                if (y < 0) y = 0;
                float pred_y = 0.0;
                for (auto field : x) {
                    pred_y += rets_w[field.fea] * field.val;
                }
                pred_y = 1. / (1. + exp(-1 * pred_y));
                for (auto field : x) {
                    delta[field.fea] += alpha * field.val * (pred_y - y);
                }
            }

            for (int i = 0; i < delta.size(); i++) {
                delta[i] = delta[i] / sum;
            }

            kvworker->Push(kv_u, keys, delta);
        }
        else {
            // Create a DataSampler for SGD
            datastore::DataIterator<LabeledPointHObj<float, float, true>> data_iterator(data_store);
            // do SGD
            // pull from kvstore_w
            std::vector<float> rets_w;
            kvworker->Wait(kv_w, kvworker->Pull(kv_w, keys, &rets_w));
            std::vector<float> old_rets_w = rets_w;
            // pull from kvstore_u
            std::vector<float> rets_u;
            kvworker->Wait(kv_u, kvworker->Pull(kv_u, keys, &rets_u));

            std::vector<float> rets_w_delta(num_params);
            while (data_iterator.has_next()) {
                // get next data 
                auto& data = data_iterator.next();

                std::vector<float> delta(num_params);
                std::vector<float> old_delta(num_params);

                auto& x = data.x;
                float y = data.y;
                if (y < 0) y = 0;
                float pred_y = 0.0;
                float old_pred_y = 0.0;
                // new gradient
                for (auto field : x) {
                    pred_y += rets_w[field.fea] * field.val;
                }
                pred_y = 1. / (1. + exp(-1 * pred_y));
                for (auto field : x) {
                    delta[field.fea] = alpha * field.val * (pred_y - y);
                }

                // old gradient
                for (auto field : x) {
                    old_pred_y = old_rets_w[field.fea] * field.val;
                }
                old_pred_y = 1. / (1. + exp(-1 * old_pred_y));
                for (auto field : x) {
                    old_delta[field.fea] = alpha * field.val * (old_pred_y - y);
                }

                // update rets_w
                // calculate rets_w_delta
                for (int i = 0; i < rets_u.size(); i++) {
                    // record rets_w_delta
                    float w_delta = (-1) * alpha * (delta[i] - old_delta[i] + rets_u[i]);
                    rets_w_delta[i] = w_delta;
                    // update rets_w
                    rets_w[i] += w_delta;
                } 
                test_error(rets_w, data_store); 
            } 
            kvworker->Push(kv_w, keys, rets_w_delta);

            husky::LOG_I << "test model";
            if (info.get_cluster_id() == 0) {
                test_error(rets_w, data_store);
            }
        }
    });

    engine.Submit();

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
