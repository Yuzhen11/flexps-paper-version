// Copyright 2016 Husky Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <fstream>
#include <string>

#include "hdfs/hdfs.h"

#include "husky/base/assert.hpp"
#include "husky/base/serialization.hpp"
#include "husky/core/constants.hpp"
#include "husky/core/context.hpp"
#include "husky/core/network.hpp"
#include "husky/io/input/hdfs_binary_inputformat.hpp"
#include "husky/io/input/binary_inputformat_impl.hpp"

#include "core/constants.hpp"

namespace husky {
namespace io {

class HDFSFileAskerML : public HDFSFileAsker {
   public:
    void init(const std::string& path, int id, int num_threads, const std::string& filter="") {
        if (!filter.empty() || path.find(':') == std::string::npos) {
            file_url_ = path + ":" + filter;
        } else {
            file_url_ = path;
        }

        id_ = id;
        num_threads_ = num_threads;
    }

    std::string fetch_new_file() {
        base::BinStream request;
        request << get_hostname() << file_url_ << id_ << num_threads_ << husky::Context::get_param("kLoadHdfsType");
        base::BinStream response = Context::get_coordinator()->ask_master(request, constants::kIOHDFSBinarySubsetLoad);
        return base::deser<std::string>(response);
    }

   private:
    std::string file_url_;
    int id_;
    int num_threads_;
};

class HDFSBinaryInputFormatML : public HDFSBinaryInputFormat {
   public:
    HDFSBinaryInputFormatML(int num_threads, int id) {
        id_ = id;
        num_threads_ = num_threads;
    }

    ~HDFSBinaryInputFormatML(){}

    void set_input(const std::string& path, const std::string& filter="") {
        asker_.init(path, id_, num_threads_, filter);
        this->to_be_setup();
    }

    // Should not used by user
    bool next(BinaryInputFormatImpl::RecordT& record) {
        std::string file_name = asker_.fetch_new_file();
        if (file_name.empty()) {
            return false;
        }
        record.set_bin_stream(new HDFSFileBinStream(fs_, file_name));
        return true;
    }

   private:
    HDFSFileAskerML asker_;
    int id_;
    int num_threads_;
};

}  // namespace io
}  // namespace husky
