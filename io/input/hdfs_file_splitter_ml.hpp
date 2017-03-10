#pragma once

#include <string>

#include "husky/io/input/hdfs_file_splitter.hpp"

#include "husky/base/exception.hpp"
#include "husky/core/coordinator.cpp"

#include "core/constants.hpp"

namespace husky {
namespace io {

class HDFSFileSplitterML : public HDFSFileSplitter {
   public:
    HDFSFileSplitterML(int num_threads, int id) {
        num_threads_ = num_threads;
        id_ = id;
    }
    virtual ~HDFSFileSplitterML() {}
        
    boost::string_ref fetch_block(bool is_next = false) {
        int nbytes = 0;
        
        if (is_next) {
            nbytes = hdfsRead(fs_, file_, data_, hdfs_block_size);
            if (nbytes == 0)
                return "";
            if (nbytes == -1) {
                throw base::HuskyException("read next block error!");
            }
        } else {
            // ask master for a new block
            BinStream question;
            question << url_ << husky::Context::get_param("hostname")
                << num_threads_ << id_ << husky::Context::get_param("kLoadHdfsType");
            BinStream answer = husky::Context::get_coordinator()->ask_master(question, constants::kIOHDFSSubsetLoad);
            std::string fn;
            answer >> fn;
            answer >> offset_;

            if (fn == "") {
                // no more files
                return "";
            }

            if (file_ != NULL) {
                int rc = hdfsCloseFile(fs_, file_);
                assert(rc == 0);
                // Notice that "file" will be deleted inside hdfsCloseFile
                file_ = NULL;
            }

            // read block
            nbytes = read_block(fn);
        }
        return boost::string_ref(data_, nbytes);
    }

    private:
        int num_threads_;
        int id_;
};

}  // namespace io
}  // namespace husky
