/* 
 * Parameters
 *
 * input
 * type: double
 * info: hdfs input file path
 *
 * num_features
 * type: int
 * info: number of features
 *
 * example:
 * input=hdfs:///1155014536/a9
 * num_feature:123
 *
 */

#include "boost/tokenizer.hpp"
#include "datastore/datastore.hpp"
#include "husky/io/input/inputformat_store.hpp"
#include "husky/lib/ml/feature_label.hpp"
#include "worker/engine.hpp"

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
                        int idx = std::stoi(*it++) - 1;
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
    bool rt = init_with_args(argc, argv, {
        "worker_port", "cluster_manager_host", "cluster_manager_port", "input", "hdfs_namenode", "hdfs_namenode_port", "num_features"
    });
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    datastore::DataStore<LabeledPointHObj<double, double, true>> data_store(Context::get_worker_info().get_num_local_workers());

    auto task = TaskFactory::Get().CreateTask<HuskyTask>(1, 4); // 1 epoch, 4 workers
    engine.AddTask(std::move(task), [&data_store](const Info& info) {
        // load
        int num_features = std::stoi(Context::get_param("num_features"));
        auto local_id = info.get_local_id();
        load_data(Context::get_param("input"), data_store, DataFormat::kLIBSVMFormat, num_features, local_id);
    });
    engine.Submit();

    auto task2 = TaskFactory::Get().CreateTask<HuskyTask>(1, 4);
    engine.AddTask(std::move(task2), [&data_store](const Info& info) {
        // read from datastore
        int counter = 0;
        auto& local_data = data_store.Pull(info.get_local_id());
        for (auto& data : local_data) {
            counter += 1;
        }
        husky::LOG_I << "record num: " << std::to_string(counter);
    });
    engine.Submit();
    engine.Exit();
}
