#pragma once

#include "husky/core/worker_info.hpp"
#include "husky/io/input/line_inputformat.hpp"

#include "io/input/hdfs_file_splitter_ml.hpp"

namespace husky {
namespace io {

class LineInputFormatML : public LineInputFormat {
   public:
    LineInputFormatML(int num_threads, int id) {
        num_threads_ = num_threads;
        id_ = id;
        splitter_ = new HDFSFileSplitterML(num_threads_, id_);
    }

    virtual ~LineInputFormatML(){}
        
    /// function for creating different splitters for different urls
    void set_splitter(const std::string& url) {
        if (!url_.empty() && url_ == url)
            // Setting with a same url last time will do nothing.
            return;
        url_ = url;

        int prefix = url_.find("://");
        ASSERT_MSG(prefix != std::string::npos, ("Cannot analyze protocol from " + url_).c_str());
        splitter_->load(url_.substr(prefix + 3));
    }

    void set_num_threads(int num_threads) {
        num_threads_ = num_threads; 
    }

    void set_worker_info(int id) {
        id_ = id;
    }
        
   private:
    int num_threads_;
    int id_;
};

}
}
