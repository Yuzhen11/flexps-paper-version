/*
References:
    1) David Arthur and Sergei Vassilvitskii.k-means++: The Advantages of Careful Seeding
    2) D. Sculley.Web-Scale K-Means Clustering

Some configuration:

    num_train_workers=4
    num_load_workers=4

    input=hdfs:///datasets/classification/a9
    num_features=123
    kLoadHdfsType=load_hdfs_globally
    learning_rate_coefficient=0.15
    K=2

    num_iters=200000
    train_epoch=2
    test_iters=500
    test_size=30000

    use_chunk=on
    batch_size=300
*/

#include <unistd.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include "datastore/datastore.hpp"
#include "kvstore/kvstore.hpp"
#include "ml/ml.hpp"
#include "worker/engine.hpp"

#include "datastore/data_sampler.hpp"
#include "husky/lib/ml/feature_label.hpp"

// for load_data()
#include "boost/tokenizer.hpp"
#include "husky/io/input/inputformat_store.hpp"
#include "io/input/line_inputformat_ml.hpp"

using namespace husky;
using husky::lib::ml::LabeledPointHObj;
enum class DataFormat { kLIBSVMFormat, kTSVFormat };

// load data evenly
template <typename FeatureT, typename LabelT, bool is_sparse>
void load_data(std::string url, datastore::DataStore<LabeledPointHObj<FeatureT, LabelT, is_sparse>>& data,
               DataFormat format, int num_features, const Info& info) {
    ASSERT_MSG(num_features > 0, "the number of features is non-positive.");
    using DataObj = LabeledPointHObj<FeatureT, LabelT, is_sparse>;

    auto local_id = info.get_local_id();
    auto num_workers = info.get_num_workers();

    // set parser1
    auto parser1 = [&data, &local_id, &num_workers, &num_features](boost::string_ref chunk) {
        if (chunk.empty())
            return;

        DataObj this_obj(num_features);

        char* pos;
        std::unique_ptr<char> chunk_ptr(new char[chunk.size() + 1]);
        strncpy(chunk_ptr.get(), chunk.data(), chunk.size());
        chunk_ptr.get()[chunk.size()] = '\0';
        char* tok = strtok_r(chunk_ptr.get(), " \t:", &pos);

        int i = -1;
        int idx;
        double val;
        while (tok != NULL) {
            if (i == 0) {
                idx = std::atoi(tok) - 1;
                i = 1;
            } else if (i == 1) {
                val = std::atof(tok);
                this_obj.x.set(idx, val);
                i = 0;
            } else {
                this_obj.y = std::atof(tok);
                i = 0;
            }
            // Next key/value pair
            tok = strtok_r(NULL, " \t:", &pos);
        }
        data.Push(local_id, std::move(this_obj));
    };

    // set parser2
    int data_count = 0;
    std::vector<husky::base::BinStream> send_buffer(num_workers);
    auto parser2 = [&send_buffer, &num_workers, &data_count](boost::string_ref& chunk) {
        if (chunk.size() == 0)
            return;
        std::string line(chunk.data(), chunk.size());
        // evenly assign docs to all threads
        int dist = data_count % num_workers;
        send_buffer[dist] << line;
        data_count++;  // every data occupies a line
    };

    // setup input format
    auto& infmt = husky::io::InputFormatStore::create_line_inputformat();
    infmt.set_input(url);

    // loading
    typename io::LineInputFormat::RecordT record;
    bool success = false;
    while (true) {
        success = infmt.next(record);
        if (success == false)
            break;
        parser2(io::LineInputFormat::recast(record));
    }

    // evenly assign docs to all threads
    auto* mailbox = Context::get_mailbox(info.get_local_id());
    for (int i = 0; i < num_workers; i++) {
        int dist = info.get_tid(i);
        if (send_buffer[i].size() == 0)
            continue;
        mailbox->send(dist, 2, 0, send_buffer[i]);  // params: dst, channel, progress, bin
    }
    mailbox->send_complete(2, 0, info.get_worker_info().get_local_tids(),
                           info.get_worker_info().get_pids());  // params: channel, progress, sender_tids, recv_tids

    while (mailbox->poll(2, 0)) {
        auto bin = mailbox->recv(2, 0);
        std::string line;
        while (bin.size() != 0) {
            bin >> line;
            parser1(line);
        }
    }
}

// return ID of cluster whose center is the nearest (uses euclidean distance)
int get_id_nearest_center(auto& point, int K, auto& params, int num_features) {
    double square_dist, min_square_dist = std::numeric_limits<double>::max();
    int id_cluster_center = -1;
    auto& x = point.x;

    for (int i = 0; i < K; i++)  // calculate the dist between point and clusters[i]
    {
        std::vector<double>::const_iterator first = params.begin() + i * num_features;
        std::vector<double>::const_iterator last = first + num_features;
        std::vector<double> diff(first, last);

        for (auto field : x)
            diff[field.fea] -= field.val;

        square_dist = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);

        if (square_dist < min_square_dist) {
            min_square_dist = square_dist;
            id_cluster_center = i;
        }
    }

    return id_cluster_center;
}

// return square distance between a point and its nearest center
template <typename T>
T get_dist_nearest_center(auto& point, int current_num_center, auto& params, int num_features) {
    double square_dist, min_square_dist = std::numeric_limits<double>::max();
    auto& x = point.x;

    for (int i = 0; i < current_num_center; i++)  // calculate the dist between point and clusters[i]
    {
        std::vector<double>::const_iterator first = params.begin() + i * num_features;
        std::vector<double>::const_iterator last = first + num_features;
        std::vector<double> diff(first, last);

        for (auto field : x)
            diff[field.fea] -= field.val;

        square_dist = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);

        if (square_dist < min_square_dist)
            min_square_dist = square_dist;
    }

    return min_square_dist;
}

int main(int argc, char* argv[]) {
    bool rt =
        init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", "hdfs_namenode",
                                    "hdfs_namenode_port", "input", "num_features", "num_iters", "train_epoch"});

    int train_epoch = std::stoi(Context::get_param("train_epoch"));
    int num_iters = std::stoi(Context::get_param("num_iters"));
    int num_features = std::stoi(Context::get_param("num_features"));
    int K = std::stoi(Context::get_param("K"));
    int batch_size = std::stoi(Context::get_param("batch_size"));
    int test_iters = std::stoi(Context::get_param("test_iters"));
    int test_size = std::stoi(Context::get_param("test_size"));
    double learning_rate_coefficient = std::stod(Context::get_param("learning_rate_coefficient"));
    int num_train_workers = std::stoi(Context::get_param("num_train_workers"));
    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));

    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    // Create the DataStore
    datastore::DataStore<LabeledPointHObj<double, int, true>> data_store(
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

    // set hint for kvstore and MLTask
    std::map<std::string, std::string> hint = {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kConsistency, husky::constants::kASP},
        {husky::constants::kNumWorkers, "1"},
    };
    int kv = kvstore::KVStore::Get().CreateKVStore<double>(hint);  // didn't set max_key and chunk_size

    // initialization task
    auto init_task = TaskFactory::Get().CreateTask<MLTask>(1, 1, Task::Type::MLTaskType);
    init_task.set_hint(hint);
    init_task.set_kvstore(kv);
    init_task.set_dimensions(K * num_features +
                             K);  // use params[K * num_features] - params[K * num_features + K] to store v[K]

    engine.AddTask(std::move(init_task), [&data_store, &K, &num_features](const Info& info) {

        // initialize a worker
        auto worker = ml::CreateMLWorker<double>(info);
        std::vector<husky::constants::Key> all_keys;
        for (int i = 0; i < K * num_features + K; i++)  // set keys
            all_keys.push_back(i);

        // read from datastore
        auto& local_data = data_store.Pull(info.get_local_id());
        husky::LOG_I << YELLOW("local_data.size: " + std::to_string(local_data.size()));
        std::vector<double> params;

        // use only one worker to initialize the params
        auto start_time = std::chrono::steady_clock::now();
        worker->Pull(all_keys, &params);

        int index_point;
        std::vector<int> prohibited_indexes;

        // K-means: choose K distinct values for the centers of the clusters randomly
        /*for (int i = 0; i < K; i++)
        {
            while (true)
            {
                srand (time(NULL));
                index_point = rand() % local_data.size();
                if (find(prohibited_indexes.begin(), prohibited_indexes.end(), index_point) == prohibited_indexes.end())
        // not found, this index_point can be used
                {
                    prohibited_indexes.push_back(index_point);
                    auto& x = local_data[index_point].x;
                    for (auto field : x)
                        params[i * num_features + field.fea] = field.val;

                    break;
                }
            }
            params[K * num_features + i] += 1;
        }*/

        // K-means++
        std::vector<double> dist(local_data.size());

        index_point = rand() % local_data.size();
        auto& x = local_data[index_point].x;
        for (auto field : x)
            params[field.fea] = field.val;

        params[K * num_features] += 1;

        double sum;
        for (int i = 1; i < K; i++) {
            sum = 0;
            for (int j = 0; j < local_data.size(); j++) {
                dist[j] = get_dist_nearest_center<double>(local_data[j], i, params, num_features);
                sum += dist[j];
            }

            sum = sum * rand() / (RAND_MAX - 1.);

            for (int j = 0; j < local_data.size(); j++) {
                sum -= dist[j];
                if (sum > 0)
                    continue;

                auto& x = local_data[j].x;
                for (auto field : x)
                    params[i * num_features + field.fea] = field.val;

                break;
            }
            params[K * num_features + i] += 1;
        }

        worker->Push(all_keys, params);

        auto end_time = std::chrono::steady_clock::now();
        auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        husky::LOG_I << YELLOW("Init time: " + std::to_string(load_time) + " ms");
    });
    engine.Submit();

    // training task
    auto training_task = TaskFactory::Get().CreateTask<MLTask>(1, num_train_workers, Task::Type::MLTaskType);
    training_task.set_hint(hint);
    training_task.set_kvstore(kv);
    training_task.set_dimensions(K * num_features + K);

    engine.AddTask(std::move(training_task), [&data_store, &num_iters, &test_iters, &K, &batch_size, &num_features,
                                              &test_size, &learning_rate_coefficient](const Info& info) {

        // initialize a worker
        auto worker = ml::CreateMLWorker<double>(info);

        std::vector<husky::constants::Key> all_keys;
        for (int i = 0; i < K * num_features + K; i++)  // set keys
            all_keys.push_back(i);

        // read from datastore
        auto& local_data = data_store.Pull(info.get_local_id());

        std::vector<double> params;

        // training task
        for (int iter = 0; iter < num_iters; iter++) {
            worker->Pull(all_keys, &params);
            std::vector<double> step_sum(params);

            // training A mini-batch
            int id_nearest_center;
            double alpha;
            int itr = rand() % local_data.size();  // random start point

            for (int i = 0; i < batch_size; ++i) {  // read a mini batch
                auto& data = local_data[itr++];
                auto& x = data.x;
                id_nearest_center = get_id_nearest_center(data, K, step_sum, num_features);
                alpha = learning_rate_coefficient / ++step_sum[K * num_features + id_nearest_center];

                std::vector<double>::const_iterator first = step_sum.begin() + id_nearest_center * num_features;
                std::vector<double>::const_iterator last = step_sum.begin() + (id_nearest_center + 1) * num_features;
                std::vector<double> c(first, last);

                for (auto field : x)
                    c[field.fea] -= field.val;

                for (int j = 0; j < num_features; j++)
                    step_sum[j + id_nearest_center * num_features] -= alpha * c[j];

                if (itr == local_data.size())
                    itr = 0;
            }

            // test model, using a9 dataset which contains only two clusters
            if (iter % test_iters == 0 && info.get_cluster_id() == 0) {
                datastore::DataSampler<LabeledPointHObj<double, int, true>> data_sampler(data_store);
                data_sampler.random_start_point();
                int c_count = 0;  // correct count
                int count1 = 0, count2 = 0;

                for (int i = 0; i < test_size; i++) {
                    auto& data = data_sampler.next();
                    auto y = data.y;
                    if (y == -1)
                        y = 0;
                    /*else if (y == 1)
                        y = 1;*/

                    int pred_y = get_id_nearest_center(data, K, params, num_features);
                    if (int(pred_y) == int(y))
                        c_count++;

                    // for test
                    if (pred_y == 0)
                        count1++;
                    else if (pred_y == 1)
                        count2++;
                }
                double accu = double(c_count) / test_size;
                husky::LOG_I << "Worker " + std::to_string(info.get_cluster_id()) + ", iter " + std::to_string(iter)
                             << ":accuracy is " << GREEN(std::to_string(accu > 0.5 ? accu : 1 - accu))
                             << " count is :" << std::to_string(test_size) << " c_count is:" << std::to_string(c_count);

                husky::LOG_I << RED("Worker " + std::to_string(info.get_cluster_id()) + ": count1: " +
                                    std::to_string(count1));
                husky::LOG_I << RED("Worker " + std::to_string(info.get_cluster_id()) + ": count2: " +
                                    std::to_string(count2));
            }

            // update params
            for (int i = 0; i < step_sum.size(); i++)
                step_sum[i] -= params[i];

            worker->Push(all_keys, step_sum);
        }
    });
    engine.Submit();
    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
