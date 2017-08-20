#include <unistd.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>  // std::pair, std::make_pair
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

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include "lib/app_config.hpp"
#include "lib/task_utils.hpp"

using namespace husky;
using husky::lib::ml::LabeledPointHObj;
enum class DataFormat { kLIBSVMFormat, kTSVFormat };

template <typename T>
void vec_to_str(const std::string& name, std::vector<T>& vec, std::stringstream& ss) {
    ss << name;
    for (auto& v : vec)
        ss << "," << v;
    ss << "\n";
}

template <typename T>
void get_stage_conf(const std::string& conf_str, std::vector<T>& vec, int num_stage) {
    std::vector<std::string> split_result;
    boost::split(split_result, conf_str, boost::is_any_of(","), boost::algorithm::token_compress_on);
    vec.reserve(num_stage);
    for (auto& i : split_result) {
        vec.push_back(std::stoi(i));
    }
    assert(vec.size() == num_stage);
}

// load data evenly
template <typename FeatureT, typename LabelT, bool is_sparse>
void load_data(std::string url, datastore::DataStore<LabeledPointHObj<FeatureT, LabelT, is_sparse>>& data,
               DataFormat format, int num_features, const Info& info) {
    ASSERT_MSG(num_features > 0, "the number of features is non-positive.");
    using DataObj = LabeledPointHObj<FeatureT, LabelT, is_sparse>;

    auto local_id = info.get_local_id();
    auto num_workers = info.get_num_workers();

    // set parse_line
    auto parse_line = [&data, local_id, num_workers, num_features](boost::string_ref chunk) {
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
        float val;
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
        distribute_datapoint(io::LineInputFormat::recast(record));
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
            parse_line(line);
        }
    }
}

// calculate the square distance between two points
float dist(auto& point1, auto& point2, int num_features) {
    std::vector<float> diff(num_features);
    auto& x1 = point1.x;
    auto& x2 = point2.x;

    for (auto field : x1)
        diff[field.fea] = field.val;

    for (auto field : x2)
        diff[field.fea] -= field.val;

    return std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
}

// return ID of cluster whose center is the nearest (uses euclidean distance), and the distance
std::pair<int, float> get_nearest_center(const LabeledPointHObj<float, int, true>& point, int K,
                                         const std::vector<float>& params, int num_features) {
    float square_dist, min_square_dist = std::numeric_limits<float>::max();
    int id_cluster_center = -1;
    auto& x = point.x;

    for (int i = 0; i < K; i++)  // calculate the dist between point and clusters[i]
    {
        typename std::vector<float>::const_iterator first = params.begin() + i * num_features;
        typename std::vector<float>::const_iterator last = first + num_features;
        std::vector<float> diff(first, last);

        for (auto field : x)
            diff[field.fea] -= field.val;

        square_dist = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);

        if (square_dist < min_square_dist) {
            min_square_dist = square_dist;
            id_cluster_center = i;
        }
    }

    return std::make_pair(id_cluster_center, min_square_dist);
}

// test the Sum of Square Error of the model
void test_error(const std::vector<float>& params, datastore::DataStore<LabeledPointHObj<float, int, true>>& data_store,
                int iter, int K, int data_size, int num_features, int cluster_id) {
    datastore::DataSampler<LabeledPointHObj<float, int, true>> data_sampler(data_store);
    float sum = 0;  // sum of square error
    std::pair<int, float> id_dist;
    std::vector<int> count(3);

    for (int i = 0; i < data_size; i++) {
        // get next data
        id_dist = get_nearest_center(data_sampler.next(), K, params, num_features);
        sum += id_dist.second;
        count[id_dist.first]++;
    }

    husky::LOG_I << "Worker " + std::to_string(cluster_id) + ", iter " + std::to_string(iter)
                 << ":Within Set Sum of Squared Errors = " << GREEN(std::to_string(sum));
    for (int i = 0; i < K; i++)  // for tuning learning rate
        husky::LOG_I << RED("Worker " + std::to_string(cluster_id) + ", count" + std::to_string(i) + ": " +
                            std::to_string(count[i]));
}

void random_init(int K, int num_features, std::vector<LabeledPointHObj<float, int, true>>& local_data,
                 std::vector<float>& params) {
    std::vector<int> prohibited_indexes;
    int index;
    for (int i = 0; i < K; i++) {
        while (true) {
            srand(time(NULL));
            index = rand() % local_data.size();
            if (find(prohibited_indexes.begin(), prohibited_indexes.end(), index) ==
                prohibited_indexes.end())  // not found, this index can be used
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

void kmeans_plus_plus_init(int K, int num_features, std::vector<LabeledPointHObj<float, int, true>>& local_data,
                           std::vector<float>& params) {
    auto X = local_data;
    std::vector<float> dist(X.size());
    int index;

    index = rand() % X.size();
    auto& x = X[index].x;
    for (auto field : x)
        params[field.fea] = field.val;

    params[K * num_features] += 1;
    X.erase(X.begin() + index);

    float sum;
    int id_nearest_center;
    for (int i = 1; i < K; i++) {
        sum = 0;
        for (int j = 0; j < X.size(); j++) {
            dist[j] = get_nearest_center(X[j], i, params, num_features).second;
            sum += dist[j];
        }

        sum = sum * rand() / (RAND_MAX - 1.);

        for (int j = 0; j < X.size(); j++) {
            sum -= dist[j];
            if (sum > 0)
                continue;

            auto& x = X[j].x;
            for (auto field : x)
                params[i * num_features + field.fea] = field.val;

            X.erase(X.begin() + j);
            break;
        }
        params[K * num_features + i] += 1;
    }
}

void kmeans_parallel_init(int K, int num_features, std::vector<LabeledPointHObj<float, int, true>>& local_data,
                          std::vector<float>& params) {
    int index;
    std::vector<LabeledPointHObj<float, int, true>> C;
    auto X = local_data;
    index = rand() % X.size();
    C.push_back(X[index]);
    X.erase(X.begin() + index);
    float square_dist, min_dist;
    /*float sum_square_dist = 0;  // original K-Means|| algorithms
    for(int i = 0; i < X.size(); i++)
        sum_square_dist += dist(C[0], X[i], num_features);
    int sample_time = log(sum_square_dist), l = 2;*/
    int sample_time = 5, l = 2;  // empiric value, sampe_time: time of sampling   l: oversample coefficient

    for (int i = 0; i < sample_time; i++) {
        // compute d^2 for each x_i
        std::vector<float> psi(X.size());

        for (int j = 0; j < X.size(); j++) {
            min_dist = std::numeric_limits<float>::max();
            for (int k = 0; k < C.size(); k++) {
                square_dist = dist(X[j], C[k], num_features);
                if (square_dist < min_dist)
                    min_dist = square_dist;
            }

            psi[j] = min_dist;
        }

        float phi = 0;
        for (int i = 0; i < psi.size(); i++)
            phi += psi[i];

        // do the drawings
        for (int i = 0; i < psi.size(); i++) {
            float p_x = l * psi[i] / phi;

            if (p_x >= rand() / (RAND_MAX - 1.)) {
                C.push_back(X[i]);
                X.erase(X.begin() + i);
            }
        }
    }

    std::vector<float> w(C.size());  // by default all are zero
    for (int i = 0; i < X.size(); i++) {
        min_dist = std::numeric_limits<float>::max();
        for (int j = 0; j < C.size(); j++) {
            square_dist = dist(X[i], C[j], num_features);
            if (square_dist < min_dist) {
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

    float sum;
    for (int i = 1; i < K; i++) {
        sum = 0;
        std::vector<float> dist(C.size());
        for (int j = 0; j < C.size(); j++) {
            dist[j] = get_nearest_center(C[j], i, params, num_features).second;
            sum += dist[j];
        }

        sum = sum * rand() / (RAND_MAX - 1.);

        for (int j = 0; j < C.size(); j++) {
            sum -= dist[j];
            if (sum > 0)
                continue;

            auto& x = C[j].x;
            for (auto field : x)
                params[i * num_features + field.fea] = field.val;

            C.erase(C.begin() + j);
            break;
        }
        params[K * num_features + i] += 1;
    }
}

void init_centers(const Info& info, int num_features, int K,
                  datastore::DataStore<LabeledPointHObj<float, int, true>>& data_store, std::string init_mode) {
    // initialize a worker
    auto worker = ml::CreateMLWorker<float>(info);
    // auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());
    std::vector<husky::constants::Key> all_keys;
    for (int i = 0; i < K * num_features + K; i++)  // set keys
        all_keys.push_back(i);

    // read from datastore
    auto& local_data = data_store.Pull(info.get_local_id());
    husky::LOG_I << YELLOW("local_data.size: " + std::to_string(local_data.size()));
    std::vector<float> params(K * num_features + K);

    // use only one worker to initialize the params
    auto start_time = std::chrono::steady_clock::now();
    worker->Pull(all_keys, &params);

    if (init_mode == "random")  // K-means: choose K distinct values for the centers of the clusters randomly
        random_init(K, num_features, local_data, params);
    else if (init_mode == "kmeans++")  // K-means++
        kmeans_plus_plus_init(K, num_features, local_data, params);
    else if (init_mode ==
             "kmeans||") {  // K-means||, reference: http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf
        kmeans_parallel_init(K, num_features, local_data, params);
    }

    husky::LOG_I << RED("params's size: " + std::to_string(params.size()));
    worker->Push(all_keys, params);
}