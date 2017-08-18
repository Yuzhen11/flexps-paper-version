#include <vector>
#include <limits>
#include <unistd.h>
#include <string>
#include <numeric>
#include <iostream>
#include "datastore/datastore.hpp"
#include "worker/engine.hpp"
#include "kvstore/kvstore.hpp"
#include "ml/ml.hpp"

#include "husky/lib/ml/feature_label.hpp"
#include "datastore/data_sampler.hpp"

// for load_data()
#include "boost/tokenizer.hpp"
#include "io/input/line_inputformat_ml.hpp"
#include "husky/io/input/inputformat_store.hpp"

#include <boost/algorithm/string/split.hpp>
#include "lib/app_config.hpp"
#include <boost/algorithm/string/classification.hpp>
#include "lib/task_utils.hpp"

using namespace husky;
using husky::lib::ml::LabeledPointHObj;
enum class DataFormat { kLIBSVMFormat, kTSVFormat };


template <typename T>
void vec_to_str(const std::string& name, std::vector<T>& vec, std::stringstream& ss) {
    ss << name;
    for (auto& v : vec) ss << "," << v;
    ss << "\n";
}


template <typename T>
void get_stage_conf(const std::string& conf_str, std::vector<T>& vec, int num_stage) {
    std::vector<std::string> split_result;
    boost::split(split_result, conf_str, boost::is_any_of(","), boost::algorithm::token_compress_on);
    vec.reserve(num_stage);
    for (auto& i : split_result) { vec.push_back(std::stoi(i)); }
    assert(vec.size() == num_stage);
}


// load data evenly
template <typename FeatureT, typename LabelT, bool is_sparse>
void load_data(std::string url, datastore::DataStore<LabeledPointHObj<FeatureT, LabelT, is_sparse>>& data, DataFormat format,
               int num_features, const Info& info){
    ASSERT_MSG(num_features > 0, "the number of features is non-positive.");
    using DataObj = LabeledPointHObj<FeatureT, LabelT, is_sparse>;

    auto local_id = info.get_local_id();
    auto num_workers = info.get_num_workers();

    // set parse_line
    auto parse_line = [&data, local_id, num_workers, num_features](boost::string_ref chunk) {
        if (chunk.empty()) return;

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

    // set distribute_datapoint
    int data_count = 0;
    std::vector<husky::base::BinStream> send_buffer(num_workers);
    auto distribute_datapoint = [&send_buffer, &num_workers, &data_count](boost::string_ref chunk) {
        if (chunk.size() == 0) 
            return;
        std::string line(chunk.data(), chunk.size());
        // evenly assign docs to all threads
        int dist = data_count % num_workers;
        send_buffer[dist] << line;
        data_count++; // every data occupies a line
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
        distribute_datapoint(io::LineInputFormat::recast(record));
    }

    // evenly assign docs to all threads
    auto* mailbox = Context::get_mailbox(info.get_local_id());
    for (int i = 0; i < num_workers; i++) {
        int dist = info.get_tid(i);
        if (send_buffer[i].size() == 0)
            continue;
        mailbox->send(dist, 2, 0, send_buffer[i]); // params: dst, channel, progress, bin
    }
    mailbox->send_complete(2, 0, 
            info.get_worker_info().get_local_tids(), info.get_worker_info().get_pids()); // params: channel, progress, sender_tids, recv_tids

    while (mailbox->poll(2, 0)) {
        auto bin = mailbox->recv(2, 0);
        std::string line;
        while (bin.size() != 0) {
            bin >> line;
            parse_line(line);
        }
    }
}


// calculate the square distance between two points
double dist(auto& point1, auto& point2, int num_features)
{
    std::vector<double> diff(num_features);
    auto& x1 = point1.x;
    auto& x2 = point2.x;

    for (auto field : x1)
        diff[field.fea] = field.val;

    for (auto field : x2)
        diff[field.fea] -= field.val;

    return std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
}


// return ID of cluster whose center is the nearest (uses euclidean distance)
static int _dummy_foobar;
template <typename T>
T get_nearest_center(const LabeledPointHObj<T,int,true>& point, int K, const std::vector<T>& params, int num_features, int& id_cluster_center = _dummy_foobar)
{
    T square_dist, min_square_dist = std::numeric_limits<T>::max();
    id_cluster_center = -1;
    auto& x = point.x;

    for (int i = 0; i < K; i++) // calculate the dist between point and clusters[i]
    {
        typename std::vector<T>::const_iterator first = params.begin() + i * num_features;
        typename std::vector<T>::const_iterator last = first + num_features;
        std::vector<T> diff(first, last);

        for (auto field : x)
            diff[field.fea] -= field.val;

        square_dist = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);

        if (square_dist < min_square_dist)
        {
            min_square_dist = square_dist;
            id_cluster_center = i;
        }
    }

    return min_square_dist;
}

// test the Sum of Square Error of the model
void test_error(const std::vector<double>& params, datastore::DataStore<LabeledPointHObj<double, int, true>>& data_store, int iter, int K, int data_size, int num_features, int cluster_id)
{
    datastore::DataSampler<LabeledPointHObj<double, int, true>> data_sampler(data_store);
    double sum = 0; // sum of square error
    int pred_y;
    std::vector<int> count(3);

    for (int i = 0; i < data_size; i++) {
        // get next data
        sum += get_nearest_center<double>(data_sampler.next(), K, params, num_features, pred_y);
        count[pred_y]++;
    }

    husky::LOG_I << "Worker " + std::to_string(cluster_id) + ", iter " + std::to_string(iter) << ":Within Set Sum of Squared Errors = " << GREEN(std::to_string(sum));
    for (int i = 0; i < K; i++)  // for tuning learning rate
        husky::LOG_I << RED("Worker " + std::to_string(cluster_id) + ", count" + std::to_string(i) + ": " + std::to_string(count[i]));
}


void init_centers(const Info& info, int num_features, int K, datastore::DataStore<LabeledPointHObj<double, int, true>>& data_store, std::string init_mode){

    // initialize a worker
    auto worker = ml::CreateMLWorker<double>(info);
    std::vector<husky::constants::Key> all_keys;
    for (int i = 0; i < K * num_features + K; i++) // set keys
        all_keys.push_back(i);

    // read from datastore
    auto& local_data = data_store.Pull(info.get_local_id());
    husky::LOG_I << YELLOW("local_data.size: " + std::to_string(local_data.size()));
    std::vector<double> params(K * num_features + K);

    // use only one worker to initialize the params
    auto start_time = std::chrono::steady_clock::now();
    worker->Pull(all_keys, &params);

    int index;
    if (init_mode == "random") // K-means: choose K distinct values for the centers of the clusters randomly
    {
        std::vector<int> prohibited_indexes;
        for (int i = 0; i < K; i++)
        {
            while (true)
            {
                srand (time(NULL));
                index = rand() % local_data.size();
                if (find(prohibited_indexes.begin(), prohibited_indexes.end(), index) == prohibited_indexes.end()) // not found, this index can be used
                {
                    prohibited_indexes.push_back(index);
                    auto& x = local_data[index].x;
                    for (auto field : x)
                        params[i * num_features + field.fea] = field.val;

                    break;
                }
            }
            params[K * num_features + i] += 1;
        }
    }
    else if (init_mode == "kmeans++")// K-means++
    {
        auto X = local_data;
        std::vector<double> dist(X.size());

        index = rand() % X.size();
        auto& x = X[index].x;
        for (auto field : x)
            params[field.fea] = field.val;

        params[K * num_features] += 1;
        X.erase(X.begin() + index);

        double sum;
        int id_nearest_center;
        for (int i = 1; i < K; i++)
        {
            sum = 0;
            for (int j = 0; j < X.size(); j++){
                dist[j] = get_nearest_center<double>(X[j], i, params, num_features);
                sum += dist[j];
            }

            sum = sum * rand() / (RAND_MAX - 1.);

            for (int j = 0; j < X.size(); j++){
                sum -= dist[j];
                if (sum > 0)
                    continue;

                auto& x = X[j].x;
                for (auto field : x)
                    params[i  * num_features + field.fea] = field.val;

                X.erase(X.begin() + j);
                break;
            }
            params[K * num_features + i] += 1;
        }
    }
    else if (init_mode == "kmeans||"){ // K-means||, reference: http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf

        std::vector<LabeledPointHObj<double, int, true>> C;
        auto X = local_data;
        index = rand() % X.size();
        C.push_back(X[index]);
        X.erase(X.begin() + index);
        double square_dist, min_dist;
        /*double sum_square_dist = 0;  // original K-Means|| algorithms
        for(int i = 0; i < X.size(); i++)
            sum_square_dist += dist(C[0], X[i], num_features);
        int sample_time = log(sum_square_dist), l = 2;*/
        int sample_time = 5, l = 2; // empiric value, sampe_time: time of sampling   l: oversample coefficient

        for (int i = 0; i < sample_time; i++)
        {
            // compute d^2 for each x_i
            std::vector<double> psi(X.size());

            for (int j = 0; j < X.size(); j++)
            {
                min_dist = std::numeric_limits<double>::max();
                for (int k = 0; k < C.size(); k++){
                    square_dist = dist(X[j], C[k], num_features);
                    if (square_dist < min_dist)
                        min_dist = square_dist;
                }

                psi[j] = min_dist;
            }

            double phi = 0;
            for(int i = 0; i < psi.size(); i++)
                phi += psi[i];

            // do the drawings
            for(int i = 0; i < psi.size(); i++)
            {
                double p_x = l * psi[i] / phi;

                if(p_x >= rand() / (RAND_MAX - 1.)) 
                {
                    C.push_back(X[i]);
                    X.erase(X.begin() + i);
                }
          }
        }

        std::vector<double> w(C.size()); // by default all are zero
        for (int i = 0; i < X.size(); i++)
        {

            min_dist = std::numeric_limits<double>::max();
            for (int j = 0; j < C.size(); j++)
            {
                square_dist = dist(X[i], C[j], num_features);
                if (square_dist < min_dist)
                {
                    min_dist = square_dist;
                    index = j;
                }
            }
            // we found the minimum index, so we increment corresp. weight
            w[index]++;
        }


        // repeat kmeans++ on C
        index = rand() % C.size();
        auto& x = C[index].x;
        for (auto field : x)
            params[field.fea] = field.val;

        params[K * num_features] += 1;
        C.erase(C.begin() + index);

        double sum;
        for (int i = 1; i < K; i++)
        {
            sum = 0;
            std::vector<double> dist(C.size());
            for (int j = 0; j < C.size(); j++){
                dist[j] = get_nearest_center<double>(C[j], i, params, num_features);
                sum += dist[j];
            }

            sum = sum * rand() / (RAND_MAX - 1.);

            for (int j = 0; j < C.size(); j++){
                sum -= dist[j];
                if (sum > 0)
                    continue;

                auto& x = C[j].x;
                for (auto field : x)
                    params[i  * num_features + field.fea] = field.val;

                C.erase(C.begin() + j);
                break;
            }
            params[K * num_features + i] += 1;
        }
    } 

    husky::LOG_I << RED("params's size: " + std::to_string(params.size()));
    worker->Push(all_keys, params);
}

int main(int argc, char *argv[])
{

    // Get configs
    config::InitContext(argc, argv, {"batch_sizes", "nums_iters", "lr_coeffs", "report_interval"});
    config::AppConfig config = config::SetAppConfigWithContext();
    auto batch_size_str = Context::get_param("batch_sizes");
    auto nums_workers_str = Context::get_param("nums_train_workers");
    auto nums_iters_str = Context::get_param("nums_iters");
    auto lr_coeffs_str = Context::get_param("lr_coeffs");
    int report_interval = std::stoi(Context::get_param("report_interval")); // test performance after each test_iters
    // Get configs for each stage
    /*std::vector<int> batch_sizes;
    std::vector<int> nums_workers;
    std::vector<int> nums_iters;
    std::vector<double> lr_coeffs;
    get_stage_conf(batch_size_str, batch_sizes, config.train_epoch);
    get_stage_conf(nums_workers_str, nums_workers, config.train_epoch);
    get_stage_conf(nums_iters_str, nums_iters, config.train_epoch);
    get_stage_conf(lr_coeffs_str, lr_coeffs, config.train_epoch);*/

    std::vector<int> batch_sizes = {1600, 800};
    std::vector<int> nums_workers = {2, 1};
    std::vector<int> nums_iters = {5000, 10000};
    std::vector<double> lr_coeffs = {0.03, 0.05};

    // Show Config
    if (Context::get_worker_info().get_process_id() == 0) {
        config::ShowConfig(config);
        std::stringstream ss;
        vec_to_str("batch_sizes", batch_sizes, ss);
        vec_to_str("nums_workers", nums_workers ,ss);
        vec_to_str("nums_iters", nums_iters ,ss);
        vec_to_str("lr_coeffs", lr_coeffs ,ss);
        husky::LOG_I << ss.str();
    }


    int num_features = std::stoi(Context::get_param("num_features"));
    int K = std::stoi(Context::get_param("K"));
    int data_size = std::stoi(Context::get_param("data_size")); // the size of the whole dataset
    int num_load_workers = std::stoi(Context::get_param("num_load_workers"));
    std::string init_mode = Context::get_param("init_mode"); // randomly initialize the k center points


    auto& engine = Engine::Get();
    // Create and start the KVStore
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());

    // Create the DataStore
    datastore::DataStore<LabeledPointHObj<double, int, true>> data_store(Context::get_worker_info().get_num_local_workers());

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
    int kv = kvstore::KVStore::Get().CreateKVStore<double>(hint, K * num_features + K);  // didn't set max_key and chunk_size
    auto init_task = TaskFactory::Get().CreateTask<MLTask>(1, 1, Task::Type::MLTaskType);
    init_task.set_hint(hint);
    init_task.set_kvstore(kv);
    init_task.set_dimensions(K * num_features + K); // use params[K * num_features] - params[K * num_features + K] to store v[K]

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
    //auto hint = config::ExtractHint(config);
    //int kv = create_kvstore_and_set_hint(hint, train_task, K * num_features + K);
    train_task.set_hint(hint);
    train_task.set_kvstore(kv);
    // TODO mutable is not safe when the lambda is used more than once
    engine.AddTask(train_task, [data_size, K, num_features, report_interval, &data_store, config, &batch_sizes, &nums_iters, &lr_coeffs, &nums_workers](const Info& info) mutable {
        int current_stage = info.get_current_epoch();
        config.num_train_workers = nums_workers[current_stage];
        assert(config.num_train_workers > 0);
        config.num_iters = nums_iters[current_stage];
        assert(batch_sizes[current_stage] % config.num_train_workers == 0);
        // get batch size and learning rate for each worker thread
        config.batch_size = batch_sizes[current_stage] / config.num_train_workers;
        config.learning_rate_coefficient = lr_coeffs[current_stage];
        if (info.get_cluster_id() == 0)
            husky::LOG_I << "Stage " << current_stage << ": " << config.num_iters << "," << config.num_train_workers << "," << config.batch_size << "," << config.learning_rate_coefficient;

        // initialize a worker
        auto worker = ml::CreateMLWorker<double>(info);

        std::vector<husky::constants::Key> all_keys;
        for (int i = 0; i < K * num_features + K; i++) // set keys
            all_keys.push_back(i);

        std::vector<double> params;
        // read from datastore
        datastore::DataSampler<LabeledPointHObj<double, int, true>> data_sampler(data_store);

        // training task
        for (int iter = 0; iter < config.num_iters; iter++)
        {
            worker->Pull(all_keys, &params);
            
            std::vector<double> step_sum(params);
            
            // training A mini-batch
            int id_nearest_center;
            double alpha;
            data_sampler.random_start_point();

            auto start_train = std::chrono::steady_clock::now();
            for (int i = 0; i < config.batch_size; ++i) {  // read 100 numbers, it should go through everything
                auto& data = data_sampler.next();
                auto& x = data.x;
                get_nearest_center<double>(data, K, step_sum, num_features, id_nearest_center);
                alpha = config.learning_rate_coefficient / ++step_sum[K * num_features + id_nearest_center];

                std::vector<double>::const_iterator first = step_sum.begin() + id_nearest_center * num_features;
                std::vector<double>::const_iterator last = step_sum.begin() + (id_nearest_center + 1) * num_features;
                std::vector<double> c(first, last);
            
                for (auto field : x)
                    c[field.fea] -= field.val;

                for (int j = 0; j < num_features; j++)
                    step_sum[j + id_nearest_center * num_features] -= alpha * c[j];
            }
            
            // test model
            if (iter % report_interval == 0 && (iter / report_interval) % config.num_train_workers == info.get_cluster_id()){
                test_error(params, data_store, iter, K, data_size, num_features, info.get_cluster_id());
                auto current_time = std::chrono::steady_clock::now();
                auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_train).count();
                husky::LOG_I << CLAY("Iter, Time: " + std::to_string(iter) + "," + std::to_string(train_time));
            }
            
            // update params
            for (int i = 0; i < K * num_features + K; i++)
                step_sum[i] -= params[i];

            worker->Push(all_keys, step_sum);
        }

    });
    start_time = std::chrono::steady_clock::now();
    engine.Submit();
    end_time = std::chrono::steady_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (Context::get_process_id() == 0)
        husky::LOG_I << "Train time: " << train_time;


    engine.Submit();
    engine.Exit();
    kvstore::KVStore::Get().Stop();
}