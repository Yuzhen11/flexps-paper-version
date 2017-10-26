#include "examples/kmeans/kmeans_helper.hpp"

int main(int argc, char* argv[]) {
    // Get configs
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port",
                                      "hdfs_namenode", "hdfs_namenode_port", "input", "num_features", 
                                      "num_iters", "staleness"});
    auto batch_size_str = Context::get_param("batch_sizes");
    auto nums_workers_str = Context::get_param("nums_train_workers");
    auto nums_iters_str = Context::get_param("nums_iters");
    auto lr_coeffs_str = Context::get_param("lr_coeffs");
    int train_epoch = std::stoi(Context::get_param("train_epoch"));
    int report_interval = std::stoi(Context::get_param("report_interval"));  // test performance after each test_iters
    int num_machine = std::stoi(Context::get_param("num_machine"));          // number of machine used
    int staleness = std::stoi(Context::get_param("staleness"));
    assert(staleness >= 0 && staleness <= 50);
    // Get configs for each stage
    std::vector<int> batch_sizes;
    std::vector<int> nums_workers;
    std::vector<int> nums_iters;
    std::vector<float> lr_coeffs;
    get_stage_conf(batch_size_str, batch_sizes, train_epoch);
    get_stage_conf(nums_workers_str, nums_workers, train_epoch);
    get_stage_conf(nums_iters_str, nums_iters, train_epoch);
    get_stage_conf(lr_coeffs_str, lr_coeffs, train_epoch);

    // Show Config
    if (Context::get_worker_info().get_process_id() == 0) {
        std::stringstream ss;
        vec_to_str("batch_sizes", batch_sizes, ss);
        vec_to_str("nums_workers", nums_workers, ss);
        vec_to_str("nums_iters", nums_iters, ss);
        vec_to_str("lr_coeffs", lr_coeffs, ss);
        husky::LOG_I << ss.str();
    }

    int num_features = std::stoi(Context::get_param("num_features"));
    int K = std::stoi(Context::get_param("K"));
    int data_size = std::stoi(Context::get_param("data_size"));  // size of the whole dataset
    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    std::string init_mode = Context::get_param("init_mode");  // random, kmeans++, kmeans||

    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    // Create the DataStore
    datastore::DataStore<LabeledPointHObj<float, int, true>> data_store(
        Context::get_worker_info().get_num_local_workers());

    // Create load_task
    auto load_task =
        TaskFactory::Get().CreateTask<Task>(1, num_load_workers);  // 1 epoch, num_load_workers workers
    engine.AddTask(std::move(load_task), [&data_store, &num_features](const Info& info) {
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, info);
        husky::LOG_I << RED("Finished Load Data!");
    });

    // submit load_task
    auto start_time = std::chrono::steady_clock::now();
    engine.Submit();
    auto end_time = std::chrono::steady_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_worker_info().get_process_id() == 0)
        husky::LOG_I << YELLOW("Load time: " + std::to_string(load_time) + " ms");

    // use params[K][0] - params[K][K-1] to store v[K], assuming num_features >= K
    int kv = kvstore::KVStore::Get().CreateKVStore<float>("ssp_add_vector", 1, staleness, K * num_features + num_features,
                                                          num_features);  // set max_key and chunk_size

    auto init_task = TaskFactory::Get().CreateTask<Task>(1, 1, Task::Type::BasicTaskType);
    TableInfo table_info1 {
        kv, K * num_features + num_features, 
        husky::ModeType::PS, 
        husky::Consistency::ASP, 
        husky::WorkerType::PSWorker, 
        husky::ParamType::IntegralType
    };

    engine.AddTask(std::move(init_task), [K, num_features, &data_store, &init_mode, table_info1](const Info& info) {
        husky::LOG_I << "Table info of init_task: " << table_info1.DebugString();
        init_centers(info, table_info1, num_features, K, data_store, init_mode);
    });

    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto init_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_worker_info().get_process_id() == 0)
        husky::LOG_I << YELLOW("Init time: " + std::to_string(init_time) + " ms");

    // Train task
    auto train_task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>();
    train_task.set_total_epoch(train_epoch);
    train_task.set_worker_num(nums_workers);
    train_task.set_worker_num_type(std::vector<std::string>(nums_workers.size(), "threads_per_worker"));
    TableInfo table_info2 {
        kv, K * num_features + num_features,
        husky::ModeType::PS, 
        husky::Consistency::SSP, 
        husky::WorkerType::PSNoneChunkWorker, 
        husky::ParamType::None,
        staleness
    };

    // TODO mutable is not safe when the lambda is used more than once
    engine.AddTask(train_task, [data_size, K, num_features, report_interval, &data_store, &batch_sizes,
                                &nums_iters, &lr_coeffs, &nums_workers, &num_machine, table_info2](const Info& info) mutable {
        
        auto start_time = std::chrono::steady_clock::now();
        // set the config for this stage
        int current_stage = info.get_current_epoch();
        int current_epoch_num_train_workers = nums_workers[current_stage];
        assert(current_epoch_num_train_workers > 0);
        int current_epoch_num_iters = nums_iters[current_stage];
        assert(batch_sizes[current_stage] % current_epoch_num_train_workers == 0);
        // get batch size and learning rate for each worker thread
        int current_epoch_batch_size =
            batch_sizes[current_stage] /
            (current_epoch_num_train_workers * num_machine);  // each machine containing num_train_workers threads
        double current_epoch_learning_rate_coefficient = lr_coeffs[current_stage];
        if (info.get_cluster_id() == 0)
            husky::LOG_I << "Stage " << current_stage << ": " << current_epoch_num_iters << "," << current_epoch_num_train_workers
                         << "," << current_epoch_batch_size << "," << current_epoch_learning_rate_coefficient;

        // initialize a worker
        auto worker = ml::CreateMLWorker<float>(info, table_info2);

        // read from datastore
        datastore::DataSampler<LabeledPointHObj<float, int, true>> data_sampler(data_store);

        std::vector<size_t> chunk_ids(K + 1);
        std::iota(chunk_ids.begin(), chunk_ids.end(), 0);
        std::vector<std::vector<float>> params(K + 1);
        std::vector<std::vector<float>> step_sums(K + 1);
        std::vector<std::vector<float>*> pull_ptrs(K + 1);
        std::vector<std::vector<float>*> push_ptrs(K + 1);

        for (int i = 0; i < K + 1; ++i) {
            params[i].resize(num_features);
            pull_ptrs[i] = &params[i];
            push_ptrs[i] = &step_sums[i];
        }

        // training task
        for (int iter = 0; iter < current_epoch_num_iters; iter++) {
            worker->PullChunks(chunk_ids, pull_ptrs);
            step_sums = params;

            // training A mini-batch
            int id_nearest_center;
            float learning_rate;
            data_sampler.random_start_point();

            for (int i = 0; i < current_epoch_batch_size; ++i) {
                auto& data = data_sampler.next();
                auto& x = data.x;
                id_nearest_center = get_nearest_center(data, K, step_sums, num_features).first;
                learning_rate = current_epoch_learning_rate_coefficient / ++step_sums[K][id_nearest_center];

                std::vector<float> deltas = step_sums[id_nearest_center];

                for (auto field : x)
                    deltas[field.fea] -= field.val;

                for (int j = 0; j < num_features; j++)
                    step_sums[id_nearest_center][j] -= learning_rate * deltas[j];
            }

            // test model each report_interval (if report_inteval = 0, dont test)
            if (report_interval > 0) {
                if (iter % report_interval == 0 &&
                    (iter / report_interval) % current_epoch_num_train_workers == info.get_cluster_id()) {
                    test_error(params, data_store, iter, K, data_size, num_features, info.get_cluster_id());
                }
            }

            // update params
            for (int i = 0; i < K + 1; ++i)
                for (int j = 0; j < num_features; ++j)
                    step_sums[i][j] -= params[i][j];

            worker->PushChunks(chunk_ids, push_ptrs);
        }
        auto end_time = std::chrono::steady_clock::now();
        auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (info.get_cluster_id() == 0){
            husky::LOG_I << "training time for epoch " << current_stage << ": " << epoch_time;
            test_error(params, data_store, current_epoch_num_iters, K, data_size, num_features, info.get_cluster_id());
        }
    });
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_process_id() == 0)
        husky::LOG_I << "Total training time: " << train_time;

    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
