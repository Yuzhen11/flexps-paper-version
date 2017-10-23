#include "examples/kmeans/kmeans_helper.hpp"

int main(int argc, char* argv[]) {
    // Get configs
    config::InitContext(argc, argv, {"batch_sizes", "nums_iters", "lr_coeffs", "report_interval"});
    config::AppConfig config = config::SetAppConfigWithContext();
    auto batch_size_str = Context::get_param("batch_sizes");
    auto nums_workers_str = Context::get_param("nums_train_workers");
    auto nums_iters_str = Context::get_param("nums_iters");
    auto lr_coeffs_str = Context::get_param("lr_coeffs");
    int report_interval = std::stoi(Context::get_param("report_interval"));  // test performance after each test_iters
    int num_machine = std::stoi(Context::get_param("num_machine"));          // number of machine used
    // Get configs for each stage
    std::vector<int> batch_sizes;
    std::vector<int> nums_workers;
    std::vector<int> nums_iters;
    std::vector<float> lr_coeffs;
    get_stage_conf(batch_size_str, batch_sizes, config.train_epoch);
    get_stage_conf(nums_workers_str, nums_workers, config.train_epoch);
    get_stage_conf(nums_iters_str, nums_iters, config.train_epoch);
    get_stage_conf(lr_coeffs_str, lr_coeffs, config.train_epoch);

    // Show Config
    if (Context::get_worker_info().get_process_id() == 0) {
        config::ShowConfig(config);
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

    auto& engine = Engine::Get();
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    // Create the DataStore
    datastore::DataStore<LabeledPointHObj<float, int, true>> data_store(
        Context::get_worker_info().get_num_local_workers());

    // Create load_task
    auto load_task =
        TaskFactory::Get().CreateTask<HuskyTask>(1, num_load_workers);  // 1 epoch, num_load_workers workers
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

    // initialization task
    auto hint = config::ExtractHint(config);
    int kv = kvstore::KVStore::Get().CreateKVStore<float>(hint, K * num_features + num_features,
                                                          num_features);  // set max_key and chunk_size
    auto init_task = TaskFactory::Get().CreateTask<Task>(1, 1, Task::Type::BasicTaskType);
    init_task.set_hint(hint);
    init_task.set_kvstore(kv);
    // use params[K][0] - params[K][K-1] to store v[K], assuming num_features >= K
    init_task.set_dimensions(K * num_features + K);

    engine.AddTask(std::move(init_task), [K, num_features, &data_store, &init_mode](const Info& info) {
        init_centers(info, num_features, K, data_store, init_mode);
    });

    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto init_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_worker_info().get_process_id() == 0)
        husky::LOG_I << YELLOW("Init time: " + std::to_string(init_time) + " ms");

    // Train task
    auto train_task = TaskFactory::Get().CreateTask<ConfigurableWorkersTask>();
    train_task.set_dimensions(K * num_features + K);
    train_task.set_total_epoch(config.train_epoch);
    train_task.set_num_workers(config.num_train_workers);
    train_task.set_worker_num(nums_workers);
    train_task.set_worker_num_type(std::vector<std::string>(nums_workers.size(), "threads_per_worker"));
    train_task.set_hint(hint);
    train_task.set_kvstore(kv);

    // TODO mutable is not safe when the lambda is used more than once
    engine.AddTask(train_task, [data_size, K, num_features, report_interval, &data_store, config, &batch_sizes,
                                &nums_iters, &lr_coeffs, &nums_workers, &num_machine](const Info& info) mutable {
        auto start_time = std::chrono::steady_clock::now();
        // set the config for this stage
        int current_stage = info.get_current_epoch();
        config.num_train_workers = nums_workers[current_stage];
        assert(config.num_train_workers > 0);
        config.num_iters = nums_iters[current_stage];
        assert(batch_sizes[current_stage] % config.num_train_workers == 0);
        // get batch size and learning rate for each worker thread
        config.batch_size =
            batch_sizes[current_stage] /
            (config.num_train_workers * num_machine);  // each machine containing num_train_workers threads
        config.learning_rate_coefficient = lr_coeffs[current_stage];
        if (info.get_cluster_id() == 0)
            husky::LOG_I << "Stage " << current_stage << ": " << config.num_iters << "," << config.num_train_workers
                         << "," << config.batch_size << "," << config.learning_rate_coefficient;

        // initialize a worker
        auto worker = ml::CreateMLWorker<float>(info);

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
        for (int iter = 0; iter < config.num_iters; iter++) {
            worker->PullChunks(chunk_ids, pull_ptrs);
            step_sums = params;

            // training A mini-batch
            int id_nearest_center;
            float learning_rate;
            data_sampler.random_start_point();

            for (int i = 0; i < config.batch_size; ++i) {
                auto& data = data_sampler.next();
                auto& x = data.x;
                id_nearest_center = get_nearest_center(data, K, step_sums, num_features).first;
                learning_rate = config.learning_rate_coefficient / ++step_sums[K][id_nearest_center];

                std::vector<float> deltas = step_sums[id_nearest_center];

                for (auto field : x)
                    deltas[field.fea] -= field.val;

                for (int j = 0; j < num_features; j++)
                    step_sums[id_nearest_center][j] -= learning_rate * deltas[j];
            }

            // test model
            if (iter % report_interval == 0 &&
                (iter / report_interval) % config.num_train_workers == info.get_cluster_id()) {
                test_error(params, data_store, iter, K, data_size, num_features, info.get_cluster_id());
            }

            // update params
            for (int i = 0; i < K + 1; ++i)
                for (int j = 0; j < num_features; ++j)
                    step_sums[i][j] -= params[i][j];

            worker->PushChunks(chunk_ids, push_ptrs);
        }
        auto end_time = std::chrono::steady_clock::now();
        auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (info.get_cluster_id() == 0)
            husky::LOG_I << "training time for epoch " << current_stage << ": " << epoch_time;
    });
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_process_id() == 0)
        husky::LOG_I << "Total training time: " << train_time;

    engine.Submit();
    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
