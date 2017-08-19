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
    // Get configs for each stage
    /*std::vector<int> batch_sizes;
    std::vector<int> nums_workers;
    std::vector<int> nums_iters;
    std::vector<float> lr_coeffs;
    get_stage_conf(batch_size_str, batch_sizes, config.train_epoch);
    get_stage_conf(nums_workers_str, nums_workers, config.train_epoch);
    get_stage_conf(nums_iters_str, nums_iters, config.train_epoch);
    get_stage_conf(lr_coeffs_str, lr_coeffs, config.train_epoch);*/

    std::vector<int> batch_sizes = {100, 300, 1000};
    std::vector<int> nums_workers = {4, 4, 4};
    std::vector<int> nums_iters = {10000, 10000, 10000};
    std::vector<float> lr_coeffs = {0.03, 0.03, 0.03};

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
    int data_size = std::stoi(Context::get_param("data_size"));  // the size of the whole dataset
    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    std::string init_mode = Context::get_param("init_mode");  // randomly initialize the k center points

    auto& engine = Engine::Get();
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    // Create the DataStore
    datastore::DataStore<LabeledPointHObj<float, int, true>> data_store(
        Context::get_worker_info().get_num_local_workers());

    // Create load_task
    auto load_task = TaskFactory::Get().CreateTask<HuskyTask>(1, num_load_workers);  // 1 epoch, 4 workers
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
    int kv =
        kvstore::KVStore::Get().CreateKVStore<float>(hint, K * num_features + K);  // didn't set max_key and chunk_size
    auto init_task = TaskFactory::Get().CreateTask<MLTask>(1, 1, Task::Type::MLTaskType);
    init_task.set_hint(hint);
    init_task.set_kvstore(kv);
    init_task.set_dimensions(K * num_features +
                             K);  // use params[K * num_features] - params[K * num_features + K] to store v[K]

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
    train_task.set_dimensions(config.num_params);
    train_task.set_total_epoch(config.train_epoch);
    train_task.set_num_workers(config.num_train_workers);
    train_task.set_worker_num(nums_workers);
    train_task.set_worker_num_type(std::vector<std::string>(nums_workers.size(), "threads_per_worker"));
    train_task.set_hint(hint);
    train_task.set_kvstore(kv);
    // TODO mutable is not safe when the lambda is used more than once
    engine.AddTask(train_task, [data_size, K, num_features, report_interval, &data_store, config, &batch_sizes,
                                &nums_iters, &lr_coeffs, &nums_workers](const Info& info) mutable {
        auto start_time = std::chrono::steady_clock::now();
        int current_stage = info.get_current_epoch();
        config.num_train_workers = nums_workers[current_stage];
        assert(config.num_train_workers > 0);
        config.num_iters = nums_iters[current_stage];
        assert(batch_sizes[current_stage] % config.num_train_workers == 0);
        // get batch size and learning rate for each worker thread
        config.batch_size = batch_sizes[current_stage] /
                            (config.num_train_workers * 5);  // use 5 workers, each containing num_train_workers threads
        config.learning_rate_coefficient = lr_coeffs[current_stage];
        if (info.get_cluster_id() == 0)
            husky::LOG_I << "Stage " << current_stage << ": " << config.num_iters << "," << config.num_train_workers
                         << "," << config.batch_size << "," << config.learning_rate_coefficient;

        // initialize a worker
        auto worker = ml::CreateMLWorker<float>(info);

        std::vector<husky::constants::Key> all_keys;
        for (int i = 0; i < K * num_features + K; i++)  // set keys
            all_keys.push_back(i);

        std::vector<float> params;
        // read from datastore
        datastore::DataSampler<LabeledPointHObj<float, int, true>> data_sampler(data_store);

        // training task
        for (int iter = 0; iter < config.num_iters; iter++) {
            worker->Pull(all_keys, &params);

            /*if (iter %1000 == 0 && current_stage == 0 && info.get_cluster_id() == 0){
                std::stringstream ss;
                for (int i = 0; i < params.size(); i++)
                    ss << params[i] << " ";

                husky::LOG_I << "iter: " + std::to_string(iter) + ", " + ss.str();
            }*/

            std::vector<float> step_sum(params);

            // training A mini-batch
            int id_nearest_center;
            float alpha;
            data_sampler.random_start_point();

            auto start_train = std::chrono::steady_clock::now();
            for (int i = 0; i < config.batch_size; ++i) {  // read 100 numbers, it should go through everything
                auto& data = data_sampler.next();
                auto& x = data.x;
                id_nearest_center = get_nearest_center(data, K, step_sum, num_features).first;
                alpha = config.learning_rate_coefficient / ++step_sum[K * num_features + id_nearest_center];

                std::vector<float>::const_iterator first = step_sum.begin() + id_nearest_center * num_features;
                std::vector<float>::const_iterator last = step_sum.begin() + (id_nearest_center + 1) * num_features;
                std::vector<float> c(first, last);

                for (auto field : x)
                    c[field.fea] -= field.val;

                for (int j = 0; j < num_features; j++)
                    step_sum[j + id_nearest_center * num_features] -= alpha * c[j];
            }

            // test model
            /*if (iter % report_interval == 0 &&
                (iter / report_interval) % config.num_train_workers == info.get_cluster_id()) {
                test_error(params, data_store, iter, K, data_size, num_features, info.get_cluster_id());
                auto current_time = std::chrono::steady_clock::now();
                auto train_time =
                    std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_train).count();
                husky::LOG_I << CLAY("Iter, Time: " + std::to_string(iter) + "," + std::to_string(train_time));
            }*/

            // update params
            for (int i = 0; i < K * num_features + K; i++)
                step_sum[i] -= params[i];

            worker->Push(all_keys, step_sum);
        }
        auto end_time = std::chrono::steady_clock::now();
        auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (Context::get_process_id() == 0)
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
