#include "examples/kmeans/kmeans_helper.hpp"

int main(int argc, char* argv[]) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port",
                                          "hdfs_namenode", "hdfs_namenode_port", "input", "num_features", "num_iters"});

    int num_iters = std::stoi(Context::get_param("num_iters"));
    int num_features = std::stoi(Context::get_param("num_features"));
    int K = std::stoi(Context::get_param("K"));
    int batch_size = std::stoi(Context::get_param("batch_size"));
    int report_interval = std::stoi(Context::get_param("report_interval"));  // test performance after each test_iters
    int data_size = std::stoi(Context::get_param("data_size"));              // the size of the whole dataset
    float learning_rate_coefficient = std::stod(Context::get_param("learning_rate_coefficient"));
    int num_train_workers = std::stoi(Context::get_param("num_train_workers"));
    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    std::string init_mode = Context::get_param("init_mode");  // randomly initialize the k center points

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

    // set hint for kvstore and Task
    std::map<std::string, std::string> hint = {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kConsistency, husky::constants::kASP},
        {husky::constants::kNumWorkers, "1"},
    };
    int kv = kvstore::KVStore::Get().CreateKVStore<float>(hint, K * num_features + num_features,
                                                          num_features);  // set max_key and chunk_size

    // initialization task
    auto init_task = TaskFactory::Get().CreateTask<Task>(1, 1, Task::Type::BasicTaskType);
    init_task.set_hint(hint);
    init_task.set_kvstore(kv);
    // use params[K][0] - params[K][K-1] to store v[K], assuming num_features >= K
    init_task.set_dimensions(K * num_features + num_features);

    engine.AddTask(std::move(init_task), [K, num_features, &data_store, &init_mode](const Info& info) {
        init_centers(info, num_features, K, data_store, init_mode);
    });

    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto init_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_worker_info().get_process_id() == 0)
        husky::LOG_I << YELLOW("Init time: " + std::to_string(init_time) + " ms");

    // training task
    auto training_task = TaskFactory::Get().CreateTask<Task>(1, num_train_workers, Task::Type::BasicTaskType);

    training_task.set_hint(hint);
    training_task.set_kvstore(kv);
    training_task.set_dimensions(K * num_features + num_features);

    engine.AddTask(std::move(training_task), [&data_store, num_iters, report_interval, K, batch_size, num_features,
                                              data_size, learning_rate_coefficient,
                                              num_train_workers](const Info& info) {

        // initialize a worker
        auto worker = ml::CreateMLWorker<float>(info);

        std::vector<size_t> chunk_ids(K + 1);
        std::iota(chunk_ids.begin(), chunk_ids.end(), 0);  // set keys
        std::vector<std::vector<float>> params(K + 1);
        std::vector<std::vector<float>> step_sums(K + 1);
        std::vector<std::vector<float>*> pull_ptrs(K + 1);
        std::vector<std::vector<float>*> push_ptrs(K + 1);

        for (int i = 0; i < K + 1; ++i) {
            params[i].resize(num_features);
            pull_ptrs[i] = &params[i];
            push_ptrs[i] = &step_sums[i];
        }

        // read from datastore
        datastore::DataSampler<LabeledPointHObj<float, int, true>> data_sampler(data_store);

        // training task
        for (int iter = 0; iter < num_iters; iter++) {
            worker->PullChunks(chunk_ids, pull_ptrs);
            step_sums = params;

            // training A mini-batch
            int id_nearest_center;
            float learning_rate;
            data_sampler.random_start_point();

            auto start_train = std::chrono::steady_clock::now();
            for (int i = 0; i < batch_size / num_train_workers; ++i) {
                auto& data = data_sampler.next();
                auto& x = data.x;
                id_nearest_center = get_nearest_center(data, K, step_sums, num_features).first;
                learning_rate = learning_rate_coefficient / ++step_sums[K][id_nearest_center];

                std::vector<float> deltas = step_sums[id_nearest_center];
                for (auto field : x)
                    deltas[field.fea] -= field.val;

                for (int j = 0; j < num_features; j++)
                    step_sums[id_nearest_center][j] -= learning_rate * deltas[j];
            }

            // test model
            if (iter % report_interval == 0 && (iter / report_interval) % num_train_workers == info.get_cluster_id()) {
                test_error(params, data_store, iter, K, data_size, num_features, info.get_cluster_id());
                auto current_time = std::chrono::steady_clock::now();
                auto train_time =
                    std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_train).count();
                husky::LOG_I << CLAY("Iter, Time: " + std::to_string(iter) + "," + std::to_string(train_time));
            }

            // update params
            for (int i = 0; i < K + 1; ++i)
                for (int j = 0; j < num_features; ++j)
                    step_sums[i][j] -= params[i][j];

            worker->PushChunks(chunk_ids, push_ptrs);
        }
    });
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_worker_info().get_process_id() == 0)
        husky::LOG_I << YELLOW("Total training time: " + std::to_string(train_time) + " ms");
    engine.Exit();
    kvstore::KVStore::Get().Stop();
}