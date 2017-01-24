#pragma once

#include <string>
#include <vector>

#include "boost/tokenizer.hpp"
#include "core/constants.hpp"
#include "husky/io/input/inputformat_store.hpp"
#include "husky/lib/ml/feature_label.hpp"

namespace husky {
namespace {

class TextBuffer {
   public:
    using BatchT = std::vector<boost::string_ref>;
    // contructor takes 4 args: hdfs path, number of lines per batch, number of batches, initialize
    TextBuffer(const std::string& url, int batch_size, int batch_num, bool _init = true) : 
        eof_(false),
        batch_size_(batch_size),
        batch_num_(batch_num) {
            // 1. set input format 
            infmt_ = &husky::io::InputFormatStore::create_line_inputformat();
            infmt_->set_input(url);
            // 2. set buffer
            buffer_ = new BatchT*[batch_num_];
            // 3. start a new thread
            if (_init) init();
    }

    // destructor: stop thread and clear buffer
    virtual ~TextBuffer() {
        batch_num_ = 0;
        load_cv_.notify_all();
        thread_->join();
        delete thread_;

        for (int i = 0; i < batch_num_; ++i) {
            delete buffer_[i];
        }
        delete buffer_;
    }

    // store batch_size_ lines in the batch and return true if success
    bool get_batch(BatchT*& batch) {
        delete batch;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (batch_count_ == 0 && !eof_) {
                get_cv_.wait(lock);  // wait for main to load data
            }

            if (eof_ && batch_count_ == 0) return false;  // no more data

            batch = buffer_[start_];
            if (++start_ >= batch_num_) start_ -= batch_num_;
            --batch_count_;
        }
        load_cv_.notify_one();  // tell main to continue loading data
        return true;
    }

    void init() {
        ASSERT_MSG(!init_, "The loading thread of TextBuffer is running!");
        thread_ = new std::thread(&TextBuffer::main, this);
        init_ = true;
    }

    // return the number of batches buffered
    int ask() {
        std::lock_guard<std::mutex> lock(mutex_);
        return batch_count_;
    }

    inline bool end_of_file() const { return eof_; }

    inline int get_batch_size() const { return batch_size_; }

   protected:
    virtual void main() {
        typename io::LineInputFormat::RecordT record;

        while (!eof_) {
            if (batch_num_ == 0) return;

            auto tmp = new BatchT;
            (*tmp).reserve(batch_size_);
            for (int i = 0; i < batch_size_; ++i) {
                if (infmt_->next(record)) {
                    (*tmp).push_back(std::move(io::LineInputFormat::recast(record)));
                } else {
                    eof_ = true;
                    break;
                }
            }

            {
                std::unique_lock<std::mutex> lock(mutex_);
                while (batch_count_ == batch_num_) {
                    if (batch_num_ == 0) return;
                    load_cv_.wait(lock);
                }
                if (!(*tmp).empty()) {
                    ++batch_count_;
                    buffer_[end_] = tmp;
                    if (++end_ >= batch_num_) end_ -= batch_num_;
                }
            }
            get_cv_.notify_one();
        }

        init_ = false;
    }

    using LineInputFormat = husky::io::LineInputFormat;

    // input
    LineInputFormat* infmt_ = nullptr;
    bool eof_;

    // buffer
    std::vector<boost::string_ref>** buffer_;
    int batch_size_;
    int batch_num_;
    int batch_count_ = 0;
    int end_ = 0;
    int start_ = 0;
    
    // thread
    std::thread* thread_ = nullptr;
    std::mutex mutex_;
    std::condition_variable load_cv_;
    std::condition_variable get_cv_;
    bool init_ = false;
};

template <typename T>
class SampleReader {
   public:
    SampleReader() = delete;
    SampleReader(int batch_size, int num_features, TextBuffer* tbf) : batch_size_(batch_size), batch_data_(batch_size), num_features_(num_features), tbf_(tbf) {
        assert(tbf->get_batch_size() == batch_size_);  // may not require equality
        for (auto s : batch_data_) {
            s = NULL;
        }
    }

    virtual ~SampleReader() {
        for (auto s : batch_data_) {
            if (s != NULL) delete s;
        }
    }

    int get_batch_size() const {
        return batch_size_;
    }

    virtual std::vector<husky::constants::Key> prepare_next_batch() {
        for (auto& s : batch_data_) {
            delete s;
            s = NULL;
        }
        index_set_.clear();
        typename TextBuffer::BatchT* raw = NULL;
        if (!tbf_->get_batch(raw)) return {};  // empty key
        int idx = 0;
        for (auto record : (*raw)) {
            parse_line(record, idx++);
        }
        return {index_set_.begin(), index_set_.end()};
    }

    const std::vector<T*>& get_data_ptrs() {
        return batch_data_;
    }

    inline bool is_empty() { return tbf_->end_of_file() && !tbf_->ask(); }

   protected:
    virtual void parse_line(const boost::string_ref& chunk, int pos) = 0;

    TextBuffer* tbf_;
    int batch_size_;
    int num_features_;
    std::vector<T*> batch_data_;
    std::set<husky::constants::Key> index_set_;
};

template <typename FeatureT, typename LabelT, bool is_sparse>
class LIBSVMSampleReader : public SampleReader<husky::lib::ml::LabeledPointHObj<FeatureT, LabelT, is_sparse>> {
   public:
    using Sample = husky::lib::ml::LabeledPointHObj<FeatureT, LabelT, is_sparse>;

    LIBSVMSampleReader(int batch_size, int num_features, TextBuffer* tbf) : SampleReader<Sample>(batch_size, num_features, tbf) {}
    
    void parse_line(const boost::string_ref& chunk, int pos) override {
        if (chunk.empty())
            return;
        boost::char_separator<char> sep(" \t");
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);

        Sample* this_obj = new Sample(this->num_features_);

        bool is_y = true;
        for (auto& w : tok) {
            if (!is_y) {
                boost::char_separator<char> sep2(":");
                boost::tokenizer<boost::char_separator<char>> tok2(w, sep2);
                auto it = tok2.begin();
                int idx = std::stoi(*it++) - 1;  // feature index from 0 to num_fea - 1
                double val = std::stod(*it++);
                this_obj->x.set(idx, val);
                this->index_set_.insert(idx);
            } else {
                this_obj->y = std::stod(w);
                is_y = false;
            }
        }

        this->batch_data_[pos] = this_obj;
    }
};

template <typename FeatureT, typename LabelT, bool is_sparse>
class TSVSampleReader : public SampleReader<husky::lib::ml::LabeledPointHObj<FeatureT, LabelT, is_sparse>> {
   public:
    using Sample = husky::lib::ml::LabeledPointHObj<FeatureT, LabelT, is_sparse>;

    TSVSampleReader(int batch_size, int num_features, TextBuffer* tbf) : SampleReader<Sample>(batch_size, num_features, tbf) {}

    void parse_line(const boost::string_ref& chunk, int pos) override {
        if (chunk.empty())
            return;
        boost::char_separator<char> sep(" \t");
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);

        Sample this_obj = new Sample(this->num_features_);

        int i = 0;
        for (auto& w : tok) {
            if (i < this->num_features_) {
                this->index_set_.insert(i);  // TODO: not necessary to store keys for dense form
                this_obj.x.set(i++, std::stod(w));
            } else {
                this_obj.y = std::stod(w);
            }
        }

        this->batch_data_[pos] = this_obj;
    }
};

}  // namespace anonymous
}  // husky