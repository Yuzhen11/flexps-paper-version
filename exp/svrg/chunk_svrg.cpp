
#include "worker/engine.hpp"
#include "exp/svrg/svrg.hpp"

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
                                 "sgd_threads_per_worker",  // The number of worker per process for fgd
                                 "sgd_overall_batch_size",                // Total batch size in sgd stage?
                                 "outer_loop",      // Number of outerloop i.e. (fgd+sgd)
                                 "inner_loop",       // Number of inner_loop i.e. number of minibatches
                                 "chunk_size",
                                 "staleness",
                                 "consistency"
                             });

    float alpha = std::stof(Context::get_param("alpha"));
    int data_size = std::stoi(Context::get_param("data_size"));
    int num_features = std::stoi(Context::get_param("num_features"));
    int num_params = num_features + 1;  // +1 for intercept

    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    int num_train_workers = std::stoi(Context::get_param("num_train_workers"));

    int chunk_size = std::stoi(Context::get_param("chunk_size"));
    int staleness = std::stoi(Context::get_param("staleness"));

    int fgd_threads_per_worker = std::stoi(Context::get_param("fgd_threads_per_worker"));
    int sgd_threads_per_worker = std::stoi(Context::get_param("sgd_threads_per_worker"));
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
    auto load_task = TaskFactory::Get().CreateTask<Task>();  // 1 epoch, 1 workers
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

    std::string storage_type = "bsp_add_vector";
    husky::Consistency consistency = husky::Consistency::BSP;
    if (Context::get_param("consistency") == "SSP") {
        consistency = husky::Consistency::SSP;
        storage_type = "ssp_add_vector";
    } else {
        LOG_I<<"Currently not supported!";
        return -1;
    } 

    int kv_w = kvstore::KVStore::Get().CreateKVStore<float>(storage_type, 1, staleness, num_params, chunk_size);
    int kv_u = kvstore::KVStore::Get().CreateKVStore<float>(storage_type, 1, staleness, num_params, chunk_size);
    TableInfo table_info_w{
        kv_w, num_params,
        husky::ModeType::PS,
        consistency,
        husky::WorkerType::PSNoneChunkWorker,
        husky::ParamType::None,
    };
    table_info_w.kStaleness = staleness;

    TableInfo table_info_u{
        kv_u, num_params,
        husky::ModeType::PS,
        consistency,
        husky::WorkerType::PSNoneChunkWorker,
        husky::ParamType::None
    };
    table_info_u.kStaleness = staleness;

    // create a configurable task and configure the worker_number in FGD stage and SGD stage
    int train_epoch = 2 * outer_loop;
    auto train_task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>(train_epoch, num_train_workers);

    std::vector<int> worker_num;
    std::vector<std::string> worker_num_type;
    // FGD stage 
    worker_num.push_back(fgd_threads_per_worker);
    worker_num_type.push_back("threads_per_worker");
    // SGD stage 
    worker_num.push_back(sgd_threads_per_worker);
    worker_num_type.push_back("threads_per_worker");
    train_task.set_worker_num(worker_num);
    train_task.set_worker_num_type(*((const std::vector<std::string>*) (&worker_num_type)));
	int FGD_total_time = 0;
	int SGD_total_time = 0;

    engine.AddTask(std::move(train_task), [&table_info_w, 
            &table_info_u, &data_store, &alpha, num_params, data_size,
            &fgd_threads_per_worker, sgd_threads_per_worker, inner_loop, &sgd_overall_batch_size,
            &FGD_total_time, &SGD_total_time, chunk_size](const Info& info) {
        auto start_time = std::chrono::steady_clock::now();

        datastore::DataStoreWrapper<LabeledPointHObj<float, float, true>> data_store_wrapper(data_store);

        std::string epoch_type;
        int current_epoch = info.get_current_epoch();
        if (current_epoch % 2 == 0) {
            epoch_type = "current_epoch: " + std::to_string(current_epoch) + " FGD";
            if (info.get_cluster_id() == 0) {
                husky::LOG_I << RED("FGD. current_epoch: " + std::to_string(current_epoch));
            }
            // do FGD
            start_time = std::chrono::steady_clock::now();
            FGD_update_chunk(data_store, alpha, num_params, info, 
                    data_size, table_info_w, table_info_u, chunk_size);

        } else {
            epoch_type = "current_epoch: " + std::to_string(current_epoch) + " SGD";
            if (info.get_cluster_id() == 0) {
                husky::LOG_I << RED("SGD. current_epoch: " + std::to_string(current_epoch));
            }
            // do SGD
            int total_threads = sgd_threads_per_worker* info.get_worker_info().get_num_processes();
            int batch_size_per_worker = std::ceil(1.0 * std::min(data_size, sgd_overall_batch_size) / total_threads);
            start_time = std::chrono::steady_clock::now();
            SGD_update_chunk(data_store, alpha, num_params, inner_loop, batch_size_per_worker, info,
                    table_info_w, table_info_u, chunk_size);

        }

        auto end_time = std::chrono::steady_clock::now();
        if (info.get_cluster_id() == 0) {
            auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
			if (current_epoch % 2 == 0) {
				FGD_total_time += train_time;
			} else {
				SGD_total_time += train_time;
			}
            husky::LOG_I << YELLOW(epoch_type + " Train time: " + std::to_string(train_time) + " ms");
        }
    });

    // submit train task
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    if (Context::get_worker_info().get_process_id() == 0) {
        auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
		auto end_to_end_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time0).count();
        LOG_I << YELLOW("Total SGD time: " + std::to_string(SGD_total_time) + " ms");
        LOG_I << YELLOW("Total FGD time: " + std::to_string(FGD_total_time) + " ms");
		LOG_I << YELLOW("Total train time: " + std::to_string(SGD_total_time * sgd_threads_per_worker + FGD_total_time * fgd_threads_per_worker) + " ms");
 		LOG_I<<YELLOW("End_to_end time: " + std::to_string(end_to_end_time) + " ms");
    }
    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
