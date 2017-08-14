#include <vector>
#include <limits>
#include <unistd.h>
#include <string>
#include <numeric>
#include <cmath>
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

using namespace husky;
using husky::lib::ml::LabeledPointHObj;
enum class DataFormat { kLIBSVMFormat, kTSVFormat };


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

    husky::LOG_I << "Worker " + std::to_string(cluster_id) + ", iter " + std::to_string(iter) << ":RMSE is " << GREEN(std::to_string(sqrt(sum/data_size)));
    for (int i = 0; i < K; i++)
        husky::LOG_I << RED("Worker " + std::to_string(cluster_id) + ", count" + std::to_string(i) + ": " + std::to_string(count[i]));
}


void init_centers(const Info& info, int num_features, int K, datastore::DataStore<LabeledPointHObj<double, int, true>>& data_store, std::string random_init){

    // initialize a worker
    auto worker = ml::CreateMLWorker<double>(info);
    std::vector<husky::constants::Key> all_keys;
    for (int i = 0; i < K * num_features + K; i++) // set keys
        all_keys.push_back(i);

    // read from datastore
    auto& local_data = data_store.Pull(info.get_local_id());
    husky::LOG_I << YELLOW("local_data.size: " + std::to_string(local_data.size()));
    std::vector<double> params;

    // use only one worker to initialize the params
    auto start_time = std::chrono::steady_clock::now();
    worker->Pull(all_keys, &params);

    int index;
    if (random_init == "on") // K-means: choose K distinct values for the centers of the clusters randomly
    {
        std::vector<int> prohibited_indexes;
        for (int i = 0; i < K; i++)
        {
            while (true)
            {
                srand (time(NULL));
                index = rand() % local_data.size();
                if (find(prohibited_indexes.begin(), prohibited_indexes.end(), index) == prohibited_indexes.end()) // not found, this index_point can be used
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
    else // K-means++
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

    husky::LOG_I << RED("params's size: " + std::to_string(params.size()));
    worker->Push(all_keys, params);
}