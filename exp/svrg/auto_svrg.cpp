#include "worker/engine.hpp"
#include "exp/svrg/svrg.hpp"
#include "exp/svrg/svrg_helper.hpp"

// *Sample setup*
//
// input=hdfs:///ml/webspam
// alpha=0.05
// num_features=16609143
// data_size=350000
// fgd_threads_per_worker=10
// sgd_overall_batch_size=30000
// outer_loop=5
using namespace husky;
using husky::lib::ml::LabeledPointHObj;

int main(int argc, char** argv) {
    auto start_time0 = std::chrono::steady_clock::now();
    // Set config
    bool rt = init_with_args(argc, argv,
                             {
                                 "alpha",
                                 "data_size",         // Need to know the number of data records for average gradient
                                 "num_features",
                                 "num_load_workers",  // Use this number of workers to load data
                                 "num_train_workers",
                                 "fgd_threads_per_worker",  // The number of worker per process for fgd
                                 "sgd_overall_batch_size",                // Total batch size in sgd stage?
                                 "outer_loop",      // Number of outerloop i.e. (fgd+sgd)
                                 "inner_loop"       // Number of inner_loop i.e. number of minibatches
                             });

    float alpha = std::stof(Context::get_param("alpha"));
    int data_size = std::stoi(Context::get_param("data_size"));
    int num_features = std::stoi(Context::get_param("num_features"));
    int num_params = num_features + 1;  // +1 for intercept

    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    int num_train_workers = std::stoi(Context::get_param("num_train_workers"));

    int threads_per_worker = std::stoi(Context::get_param("fgd_threads_per_worker"));
    int sgd_overall_batch_size = std::stoi(Context::get_param("sgd_overall_batch_size"));
    int outer_loop = std::stoi(Context::get_param("outer_loop"));
    int inner_loop = std::stoi(Context::get_param("inner_loop"));

    auto& engine = Engine::Get();
    // start kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());
    // create DataStore
    datastore::DataStore<LabeledPointHObj<float, float, true>> data_store(Context::get_worker_info().get_num_local_workers());

    // create and submit load_task
    auto load_task = TaskFactory::Get().CreateTask<Task>(); 
    load_task.set_num_workers(num_load_workers);
    engine.AddTask(std::move(load_task), [&data_store, &num_features](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, local_id);
        datastore::DataStoreWrapper<LabeledPointHObj<float, float, true>> data_store_wrapper(data_store);
        husky::LOG_I << YELLOW("datasize: " + std::to_string(data_store_wrapper.get_data_size()));

    });
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_worker_info().get_process_id() == 0)
        husky::LOG_I << YELLOW("Load time: " + std::to_string(load_time) + " ms");

    int kv_w = kvstore::KVStore::Get().CreateKVStore<float>("bsp_add_vector", threads_per_worker, -1, num_params);  // for bsp server
    int kv_u = kvstore::KVStore::Get().CreateKVStore<float>("bsp_add_vector", threads_per_worker, -1, num_params);

    // new version begins //
	int FGD_total_time = 0;
	int SGD_total_time = 0;
    // Define FGD_task
    auto FGD_task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>(1, num_train_workers);
    FGD_task.set_worker_num({threads_per_worker});
    FGD_task.set_worker_num_type({"threads_per_worker"});
    auto FGD_lambda = [kv_w, kv_u, &data_store, alpha, num_params, data_size, &FGD_total_time] (const Info& info) { 
        auto start_time = std::chrono::steady_clock::now();

        FGD_update(data_store, kv_w, kv_u, alpha, num_params, info, data_size);

        auto end_time = std::chrono::steady_clock::now();
        if (info.get_cluster_id() == 0) {
            auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
			FGD_total_time += train_time;
            husky::LOG_I << YELLOW("FGD Train time: " + std::to_string(train_time) + " ms");
        }
    };
    
    // Define SGD_task
    auto SGD_task = TaskFactory::Get().CreateTask<AutoParallelismTask>();
    SGD_task.set_epoch_iters_and_batchsizes({inner_loop}, {sgd_overall_batch_size});
    SGD_task.set_epoch_lambda([kv_w, kv_u, &data_store, alpha, num_params,
            data_size, &SGD_total_time, sgd_overall_batch_size] (const Info& info, int inner_loop) {
        auto start_time = std::chrono::steady_clock::now();
        int batch_size_per_worker = sgd_overall_batch_size * 1.0 / info.get_num_workers();
        SGD_update(data_store, kv_w, kv_u, alpha, num_params, inner_loop, batch_size_per_worker, info);

        auto end_time = std::chrono::steady_clock::now();

        if (info.get_cluster_id() == 0) {
            auto train_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
			SGD_total_time += train_time;
            husky::LOG_I << YELLOW("SGD Train time: " + std::to_string(train_time) + " ms");
            husky::LOG_I << "inner loop is:" << inner_loop;
        }
    });
   
    // Two kinds of alternating tasks FGD_task and SGD_task
    for (int i = 0; i < outer_loop; i++) {
        engine.AddTask(FGD_task, FGD_lambda); 
        engine.AddTask(SGD_task, [](const Info& info) {/*dummy lambda*/});

        // submit train task
        start_time = std::chrono::steady_clock::now();
        engine.Submit();
        end_time = std::chrono::steady_clock::now();
        if (Context::get_worker_info().get_process_id() == 0) {
            auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	    	auto end_to_end_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time0).count();
            LOG_I << YELLOW("Total SGD time: " + std::to_string(SGD_total_time) + " ms");
            LOG_I << YELLOW("Total FGD time: " + std::to_string(FGD_total_time) + " ms");
            LOG_I<< "Train time:" << train_time;
 	    	LOG_I<<YELLOW("End_to_end time: " + std::to_string(end_to_end_time) + " ms");
        }
    }
    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
