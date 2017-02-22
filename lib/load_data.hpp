#pragma once

#include "boost/tokenizer.hpp"

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
       case DataFormat::kTSVFormat: {
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

}  // namespace anonymous
}  // namespace husky
