#include <vector>

#include "boost/tokenizer.hpp"

#include "datastore/datastore.hpp"
#include "husky/io/input/inputformat_store.hpp"
#include "husky/lib/ml/feature_label.hpp"
#include "worker/engine.hpp"
#include "ml/common/mlworker.hpp"

using namespace husky;
using husky::lib::ml::LabeledPointHObj;

enum DataFormat { kLIBSVMFormat, kTSVFormat };

template <typename FeatureT, typename LabelT, bool is_sparse>
void load_data(std::string url, datastore::DataStore<LabeledPointHObj<FeatureT, LabelT, is_sparse>>& data, DataFormat format, int num_features, int local_id) {
    ASSERT_MSG(num_features > 0, "the number of features is non-positive.");
    using DataObj = LabeledPointHObj<FeatureT, LabelT, is_sparse>;

    switch(format) {
        case kLIBSVMFormat: {
            load_line_input(url, [&](boost::string_ref chunk) {
                if (chunk.empty())
                    return;
                boost::char_separator<char> sep(" \t");
                boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);

                DataObj this_obj(num_features);

                bool is_y = true;
                for (auto& w : tok) {
                    if (!is_y) {
                        boost::char_separator<char> sep2(":");
                        boost::tokenizer<boost::char_separator<char>> tok2(w, sep2);
                        auto it = tok2.begin();
                        int idx = std::stoi(*it++) - 1;// feature index from 0 to num_fea - 1
                        double val = std::stod(*it++);
                        this_obj.x.set(idx, val);
                    } else {
                        this_obj.y = std::stod(w);
                        is_y = false;
                    }
                }
                data.Push(local_id, std::move(this_obj));
            });
            break;
       }
       case kTSVFormat: {
            load_line_input(url, [&](boost::string_ref chunk) {
                if (chunk.empty())
                    return;
                boost::char_separator<char> sep(" \t");
                boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);

                DataObj this_obj(num_features);

                int i = 0;
                for (auto& w : tok) {
                    if (i < num_features) {
                        this_obj.x.set(i++, std::stod(w));
                    } else {
                        this_obj.y = std::stod(w);
                    }
                }
                data.Push(local_id, std::move(this_obj));
            });
            break;
       }
       default:
            throw base::HuskyException("Unknown data type!");
    }
}

template <typename ParseT>
void load_line_input(std::string& url, ParseT parse) {
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
        parse(io::LineInputFormat::recast(record));
    }
}

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port", 
                                       "hdfs_namenode", "hdfs_namenode_port",
                                       "input", "num_features", "alpha", "num_iters"});

    float alpha = std::stof(Context::get_param("alpha"));
    int num_iters = std::stoi(Context::get_param("num_iters"));
    int num_features = std::stoi(Context::get_param("num_features"));
    int num_params = num_features + 1; // +1 for intercept
    
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());
    datastore::DataStore<LabeledPointHObj<double, double, true>> data_store(1);

    auto task = TaskFactory::Get().CreateTask<HuskyTask>(1, 1); // 1 epoch, 1 workers
    engine.AddTask(std::move(task), [&data_store, &num_features](const Info& info) {
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, local_id);
    });


    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>();
    auto task1 = TaskFactory::Get().CreateTask<GenericMLTask>();
    task1.set_dimensions(num_params);
    task1.set_kvstore(kv1);
    //task1.set_running_type(Task::Type::HogwildTaskType);
    //task1.set_running_type(Task::Type::PSTaskType);
    task1.set_running_type(Task::Type::SingleTaskType);
    engine.AddTask(std::move(task1), [&data_store, num_iters, alpha, num_params](const Info& info) {
        auto& training_data = data_store.Pull(info.get_local_id()); //since this is a single thread task
        auto& worker = info.get_mlworker();
        std::vector<int> all_keys;
        for (int i = 0; i < num_params; i++) all_keys.push_back(i);
        std::vector<float> params(num_params, 0.1);
        for(int iter = 0; iter < num_iters; iter++) {
            worker->Pull(all_keys, &params);

            // A full batch gradient descent 
            // param += alpha * (y[i] - h(i)) * x
            std::vector<float> step_sum(num_params, 0);
            // calculate accumulated gradient
            for (auto data : training_data) {
                auto x = data.x;
                auto y = data.y;
                if (y < 0) y = 0;
                float pred_y = 0.0;
                for (auto field : x) {
                    pred_y += params[field.fea] * field.val;
                }
                pred_y += params[num_params - 1]; // intercept
                pred_y = 1. / (1. + exp(-1 * pred_y)); 

                for (auto field : x) {
                    step_sum[field.fea] += alpha * field.val * (y - pred_y);
                }
                step_sum[num_params - 1] += alpha * (y - pred_y); // intercept
            }
            // test model
            int count = 0;
            float c_count = 0;//correct count
            for (auto data : training_data) {
                count = count + 1;
                auto x = data.x;
                auto y = data.y;
                if(y < 0) y = 0;
                float pred_y = 0.0;

                for (auto field : x) {
                    pred_y += params[field.fea] * field.val;
                }
                pred_y += params[num_params - 1];
                pred_y = 1. / (1. + exp(-pred_y));
                pred_y = (pred_y > 0.5) ? 1 : 0;
                if (int(pred_y) == int(y)) { c_count += 1;}
            }
            husky::LOG_I<<std::to_string(iter)<< ":accuracy is " << std::to_string(c_count/count)<<" count is :"<<std::to_string(count)<<" c_count is:"<<std::to_string(c_count);
            // update params
            for (int j = 0; j < num_params; j++) {
                params[j] += step_sum[j]/float(count);
            }

            worker->Push(all_keys, params); 
        }

    });
    engine.Submit();
    engine.Exit();
    kvstore::KVStore::Get().Stop();
}
