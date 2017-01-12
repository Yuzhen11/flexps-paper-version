#include <vector>

#include "datastore/datastore.hpp"
#include "worker/engine.hpp"
#include "ml/common/mlworker.hpp"

#include "lib/load_data.hpp"
#include "lib/data_sampler.hpp"

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", 
                                       "hdfs_namenode", "hdfs_namenode_port",
                                       "input", "num_features", "alpha", "num_iters",
                                       "train_epoch"});

    int train_epoch = std::stoi(Context::get_param("train_epoch"));
    float alpha = std::stof(Context::get_param("alpha"));
    int num_iters = std::stoi(Context::get_param("num_iters"));
    int num_features = std::stoi(Context::get_param("num_features"));
    int num_params = num_features + 1; // +1 for intercept
    
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());
    // Create the DataStore
    datastore::DataStore<LabeledPointHObj<double, double, true>> data_store(Context::get_worker_info().get_num_local_workers());

    auto task = TaskFactory::Get().CreateTask<HuskyTask>(1, 1); // 1 epoch, 1 workers
    engine.AddTask(std::move(task), [&data_store, &num_features](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, local_id);
    });


    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
    auto task1 = TaskFactory::Get().CreateTask<GenericMLTask>();
    task1.set_dimensions(num_params);
    task1.set_kvstore(kv1);
    task1.set_total_epoch(train_epoch);
    //task1.set_running_type(Task::Type::HogwildTaskType);
    //task1.set_running_type(Task::Type::PSTaskType);
    task1.set_running_type(Task::Type::SingleTaskType);
    engine.AddTask(std::move(task1), [&data_store, num_iters, alpha, num_params](const Info& info) {
        auto& training_data = data_store.Pull(info.get_local_id()); //since this is a single thread task
        if (training_data.empty())
            return;
        auto& worker = info.get_mlworker();
        DataSampler<LabeledPointHObj<double, double, true>> data_sampler(data_store);
        data_sampler.random_start_point();
        for (int iter = 0; iter < num_iters; ++ iter) {
            auto& data = data_sampler.next();  // get next data
            auto& x = data.x;
            float y = data.y;
            if (y < 0) y = 0;
            std::vector<int> keys;
            std::vector<float> params;
            std::vector<float> delta;
            keys.reserve(x.get_feature_num()+1);
            delta.reserve(x.get_feature_num()+1);
            for (auto field : x) {  // set keys
                keys.push_back(field.fea);
            }

            worker->Pull(keys, &params);  // issue Pull

            // SGD
            float pred_y = 0.0;
            int i = 0;
            for (auto field : x) {
                pred_y += params[i++] * field.val;
            }
            pred_y = 1. / (1. + exp(-1 * pred_y)); 

            for (auto field : x) {
                delta.push_back(alpha * field.val * (y - pred_y));
            }

            worker->Push(keys, delta);  // issue Push

            // test model
            std::vector<int> all_keys;
            for (int i = 0; i < num_params; i++) all_keys.push_back(i);
            std::vector<float> test_params;
            worker->Pull(all_keys, &test_params);
            int count = 0;
            float c_count = 0;//correct count
            for (auto& data : training_data) {
                count = count + 1;
                auto x = data.x;
                auto y = data.y;
                if(y < 0) y = 0;
                float pred_y = 0.0;

                for (auto field : x) {
                    pred_y += test_params[field.fea] * field.val;
                }
                // pred_y += test_params[num_params - 1];
                pred_y = 1. / (1. + exp(-pred_y));
                pred_y = (pred_y > 0.5) ? 1 : 0;
                if (int(pred_y) == int(y)) { c_count += 1;}
            }
            husky::LOG_I<<std::to_string(iter)<< ":accuracy is " << std::to_string(c_count/count)<<" count is :"<<std::to_string(count)<<" c_count is:"<<std::to_string(c_count);
        }
    });
    engine.Submit();
    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
