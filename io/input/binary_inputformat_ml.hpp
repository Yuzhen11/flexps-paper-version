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

#include <string>

#include "husky/base/assert.hpp"
#include "husky/io/input/binary_inputformat_impl.hpp"
#include "husky/io/input/binary_inputformat.hpp"
#include "io/input/hdfs_binary_inputformat_ml.hpp"

namespace husky {
namespace io {

class BinaryInputFormatML : public BinaryInputFormat {
   public:
    typedef BinaryInputFormatImpl::RecordT RecordT;
    typedef BinaryInputFormatImpl::CastRecordT CastRecordT;

    BinaryInputFormatML(const std::string& url, int num_threads, int id, const std::string& filter=""):BinaryInputFormat(url, filter) {
        int first_colon = url.find("://");
        ASSERT_MSG(first_colon != std::string::npos, ("Cannot analyze protocol from " + url).c_str());
        std::string protocol = url.substr(0, first_colon);
        if (protocol == "hdfs") {
            infmt_impl_ = new HDFSBinaryInputFormatML(num_threads, id);
            infmt_impl_->set_input(url.substr(first_colon + 3), filter);
        } else {
            ASSERT_MSG(false, ("Unknown protocol given to BinaryInputFormat: " + protocol).c_str());
        }
    }

    BinaryInputFormatML(const BinaryInputFormat&) = delete;
    ~BinaryInputFormatML(){}
};

}  // namespace io
}  // namespace husky
