#pragma once

#include "boost/tokenizer.hpp"

#include "datastore/datastore.hpp"
#include "husky/io/input/inputformat_store.hpp"
#include "husky/lib/ml/feature_label.hpp"

#include "io/input/line_inputformat_ml.hpp"

namespace husky {
namespace {

using husky::lib::ml::LabeledPointHObj;

enum class DataFormat { kLIBSVMFormat, kTSVFormat };

template <typename FeatureT, typename LabelT, bool is_sparse>
void load_data(std::string url, datastore::DataStore<LabeledPointHObj<FeatureT, LabelT, is_sparse>>& data, DataFormat format, int num_features, int local_id) {
    ASSERT_MSG(num_features > 0, "the number of features is non-positive.");
    using DataObj = LabeledPointHObj<FeatureT, LabelT, is_sparse>;

    switch(format) {
        case DataFormat::kLIBSVMFormat: {
            load_line_input(url, [&](boost::string_ref chunk) {
                if (chunk.empty()) return;

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
            });
            break;
       }
       case DataFormat::kTSVFormat: {
            load_line_input(url, [&](boost::string_ref chunk) {
                if (chunk.empty()) return;

                DataObj this_obj(num_features);

                char* pos;
                std::unique_ptr<char> chunk_ptr(new char[chunk.size() + 1]);
                strncpy(chunk_ptr.get(), chunk.data(), chunk.size());
                chunk_ptr.get()[chunk.size()] = '\0';
                char* tok = strtok_r(chunk_ptr.get(), " \t", &pos);

                int i = 0;
                while (tok != NULL) {
                    if (i < num_features) {
                        this_obj.x.set(i++, std::stol(tok));
                    } else {
                        this_obj.y = std::stol(tok);
                    }
                    // Next key/value pair
                    tok = strtok_r(NULL, " \t", &pos);
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

}  // namespace anonymous
}  // namespace husky
