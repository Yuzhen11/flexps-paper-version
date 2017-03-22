#include <vector>

#include "datastore/datastore.hpp"
#include "datastore/datastore_utils.hpp"
#include "worker/engine.hpp"
#include "kvstore/kvstore.hpp"
#include "core/color.hpp"

#include "lib/load_data.hpp"
#include "lib/app_config.hpp"

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

/*
 * SVRG example
 */
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
            // calculate predict y
            pred_y += rets_w[field.fea] * field.val;
            // if (flag) { debug_kvstore_w += "__" + std::to_string(rets_w[field.fea]); }
        }
        pred_y = 1. / (1. + exp(-1 * pred_y));

        pred_y = (pred_y > 0.5) ? 1 : 0;
        if (int(pred_y) == int(y)) { c_count += 1;}
        flag = 0;
    }

    // husky::LOG_I << "current w: " + debug_kvstore_w;
    husky::LOG_I << ":accuracy is " << std::to_string(c_count / count) 
        << " count is :" << std::to_string(count) << " c_count is:" << std::to_string(c_count);
}

void SGD_update(datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store,
    int kv_w, int kv_u, float alpha, int num_params, int num_iters, const Info & info) {
    auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());

    // Create a DataSampler for SGD
    datastore::DataIterator<LabeledPointHObj<float, float, true>> data_iterator(data_store);

    // do SGD
    // init keys
    std::vector<husky::constants::Key> keys;
    for(int i = 0; i < num_params; i++) {
        keys.push_back(i);
    }
    // pull from kvstore_w
    std::vector<float> rets_w;
    kvworker->Wait(kv_w, kvworker->Pull(kv_w, keys, &rets_w));
    // keep old parameters
    std::vector<float> old_rets_w = rets_w;
    // pull from kvstore_u
    std::vector<float> rets_u;
    kvworker->Wait(kv_u, kvworker->Pull(kv_u, keys, &rets_u));

    std::vector<float> rets_w_delta(num_params);
    for(int i = num_iters; i > 0&&data_iterator.has_next(); i--) {
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
        // each data will calculate a delta, once a delta has been calculated, rets_w will be updated
        // so there is no need to accumulate
        for (auto field : x) {
            delta[field.fea] = alpha * field.val * (pred_y - y);
        }

        // old gradient
        for (auto field : x) {
            old_pred_y += old_rets_w[field.fea] * field.val;
        }
        old_pred_y = 1. / (1. + exp(-1 * old_pred_y));
        for (auto field : x) {
            old_delta[field.fea] = alpha * field.val * (old_pred_y - y);
        }

        // update rets_w for next calculate, but not push to kvstore, 
        for (int i = 0; i < rets_u.size(); i++) {
            // record rets_w_delta
            float w_delta = (-1) * alpha * (delta[i] - old_delta[i] + rets_u[i]);
            rets_w_delta[i] = w_delta;
            // update rets_w
            rets_w[i] += w_delta;
        }

        if (info.get_cluster_id() == 0 && i % 100 == 0) {
            test_error(rets_w, data_store);
        }
    }

    kvworker->Push(kv_w, keys, rets_w_delta);
}

void FGD_update(datastore::DataStore<LabeledPointHObj<float, float, true>>& data_store,
    int kv_w, int kv_u, float alpha, int num_params, const Info & info) {
    // when do FGD, we hope data can be loaded balancely
    // get the current_epoch
    int current_epoch = info.get_current_epoch();
    // get configurable_task's worker_num vector, worker_num vector describes how many threads each epoch will use
    std::vector<int> worker_num = static_cast<const ConfigurableWorkersTask*>(info.get_task())->get_worker_num();
    // get all threads in local process
    std::vector<int> tids = info.get_worker_info().get_local_tids(); 
    // find the pos of local_id
    int position;
    for(int k = 0; k < tids.size(); k++) {
        // get the position of current thread 
        // the position info will be used fo loading data balancely
        if (tids[k] == info.get_local_id()) {
            position = k;
            break; 
        }
    }
    // dataloadbalance
    datastore::DataLoadBalance<LabeledPointHObj<float, float, true>> data_load_balance(data_store,
        worker_num[current_epoch], position);

    // init keys
    std::vector<husky::constants::Key> keys;
    for(int i = 0; i < num_params; i++) {
        keys.push_back(i);
    }

    auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());

    // do FGD
    // pull from kvstore_w
    std::vector<float> rets_w;
    // wait util parameters have been pulled, store into rets_w
    kvworker->Wait(kv_w, kvworker->Pull(kv_w, keys, &rets_w));

    std::vector<float> delta(num_params);
    // count the num of data
    float count = 0;
    // go through all data
    while(data_load_balance.has_next()) {
        count++;
        // get the data
        auto& data = data_load_balance.next();
        auto& x = data.x;
        float y = data.y;
        float pred_y = 0.0;

        if (y < 0) y = 0;

        for(auto field : x) {
            pred_y += rets_w[field.fea] * field.val;
        }
        pred_y = 1. / (1. + exp(-1 * pred_y));

        // calculate GD
        for(auto field : x) {
            // calculate a global delta after going through all data
            // so we need to accumulate all delta 
            delta[field.fea] += alpha * field.val * (pred_y - y);
        }
    }

    // calculate avg delta to push to kvstore
    for (int i = 0; i < delta.size(); i++) {
        delta[i] = delta[i] / count;
    }

    // push kv_u
    kvworker->Push(kv_u, keys, delta);
}

int main(int argc, char** argv) {
    // Set config
    config::InitContext(argc, argv);
    auto config = config::SetAppConfigWithContext();
    if (Context::get_worker_info().get_process_id() == 0)
        config:: ShowConfig(config);
    auto hint = config::ExtractHint(config);

    int train_epoch = std::stoi(Context::get_param("train_epoch"));
    float alpha = std::stof(Context::get_param("alpha"));
    int num_iters = std::stoi(Context::get_param("num_iters"));
    int num_features = std::stoi(Context::get_param("num_features"));
    int num_params = num_features + 1; // +1 for intercept

    auto& engine = Engine::Get();
    // start the kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    // create the DataStore
    datastore::DataStore<LabeledPointHObj<float, float, true>> data_store(Context::get_worker_info().get_num_local_workers());

    // create load_task
    auto load_task = TaskFactory::Get().CreateTask<HuskyTask>(1, 1); // 1 epoch, 1 workers
    engine.AddTask(std::move(load_task), [&data_store, &num_features](const Info & info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, local_id);
        husky::LOG_I << RED("Finished Load Data!");
    });
    
    // submit load_task
    auto start_time = std::chrono::steady_clock::now(); 
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_worker_info().get_process_id() == 0)
        husky::LOG_I << YELLOW("Load time: " + std::to_string(load_time) + " ms");
    
    // exp config

    // exp 1: in SGD epoch, run single train thread
    // create train_task
    auto train_task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>(3, 6);
    train_task.set_worker_num({6, 1, 1});
    train_task.set_worker_num_type({"threads_per_worker", "threads_traverse_cluster", "threads_traverse_cluster"});
    // set svrg_type
    // type_type = generic means in SGD epoch, run multiple train threads 
    // otherwise, run single train threads
    std::string svrg_type = "";

    // exp 2: in SGD epoch, run multiple train threads
    // create train_task
    // auto train_task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>(3, 6);
    // train_task.set_worker_num({6, 3, 3});
    // // threads_per_worker means there will be 10 threads each worker, the total is 10 * num_worker 
    // // threads_per_cluster means there will be total 10 threads in the cluster
    // train_task.set_worker_num_type({"threads_per_worker", "threads_per_cluster", "threads_per_cluster"});
    // // set svrg_type
    // // type_type = generic means in SGD epoch, run multiple train threads 
    // // otherwise, run single train threads
    // std::string svrg_type = "generic";


    int kv_w = kvstore::KVStore::Get().CreateKVStore<float>();
    int kv_u = kvstore::KVStore::Get().CreateKVStore<float>();

    engine.AddTask(std::move(train_task), [&kv_w, &kv_u, &data_store, num_iters, alpha, num_params, &svrg_type](const Info & info) {
        // create a DataStoreWrapper
        datastore::DataStoreWrapper<LabeledPointHObj<float, float, true>> data_store_wrapper(data_store);
        if (data_store_wrapper.get_data_size() == 0) {
            return;  // return if there is no data
        }

        int current_epoch = info.get_current_epoch();
        // get configurable_task's worker_num vector, worker_num vector describes how many threads each epoch will use
        std::vector<int> worker_num = static_cast<const ConfigurableWorkersTask*>(info.get_task())->get_worker_num();
        
        if (current_epoch % worker_num.size() == 0) {
            // do FGD
            FGD_update(data_store, kv_w, kv_u, alpha, num_params, info);  
        } else {
            // do SGD
            if (svrg_type == "generic") {
                husky::LOG_I << RED("SGD generic");
                // generic type means there will be total worker_num[current] in cluster, we want them each thread will load 
                // num_iters / num thread to compare with exp1 fairly
                SGD_update(data_store, kv_w, kv_u, alpha, num_params, num_iters / worker_num[current_epoch], info);
            } else {
                husky::LOG_I << RED("SGD not generic");
                SGD_update(data_store, kv_w, kv_u, alpha, num_params, num_iters, info);
            }
        }
    });

    // submit train task
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count();
    husky::LOG_I << YELLOW("Train time: " + std::to_string(train_time) + " ms");

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}