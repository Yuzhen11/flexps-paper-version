#pragma once

#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "boost/tokenizer.hpp"
#include "core/constants.hpp"
#include "husky/base/log.hpp"
#include "husky/io/input/line_inputformat.hpp"
#include "husky/lib/ml/feature_label.hpp"
#include "io/input/line_inputformat_ml.hpp"
#include "core/color.hpp"

namespace husky {
namespace {

class AsyncReadBuffer final {
   public:
    using BatchT = std::vector<std::string>;

    /* 
     * contructor takes 2 args: number of lines per batch, number of batches
     */
    AsyncReadBuffer() = default;

    AsyncReadBuffer(const AsyncReadBuffer&) = delete;
    AsyncReadBuffer& operator=(const AsyncReadBuffer&) = delete;
    AsyncReadBuffer(AsyncReadBuffer&&) = delete;
    AsyncReadBuffer& operator=(AsyncReadBuffer&&) = delete;

    // destructor: stop thread and clear buffer
    ~AsyncReadBuffer() {
        batch_num_ = 0;
        load_cv_.notify_all();
        get_cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    /*
     * Function to initialize the reader threads,
     * the first thread will do the initialization
     *
     * \param url the file url in hdfs
     * \param task_id identifier to this running task
     * \param num_threads the number of worker threads we are using
     * \param batch_size the size of each batch
     * \param batch_num the number of each batch
     */
    void init(const std::string& url, int task_id, int num_threads, int batch_size, int batch_num) {
        if (init_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        if (init_) return;

        // The initialization work
        batch_size_ = batch_size;
        batch_num_ = batch_num;
        buffer_.resize(batch_num);
        infmt_.reset(new io::LineInputFormatML(num_threads, task_id));
        infmt_->set_input(url);
        thread_ = std::thread(&AsyncReadBuffer::main, this);
        init_ = true;
    }

    // store batch_size_ lines in the batch and return true if success
    bool get_batch(BatchT& batch) {
        assert(init_ == true);
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (eof_ && batch_count_ == 0) return false;  // no more data

            while (batch_count_ == 0 && !eof_) {
                get_cv_.wait(lock);  // wait for main to load data
            }

            if (eof_ && batch_count_ == 0) return false;  // no more data
            batch = std::move(buffer_[start_]);
            if (++start_ >= batch_num_) start_ -= batch_num_;
            --batch_count_;
        }
        load_cv_.notify_one();  // tell main to continue loading data
        return true;
    }

    // return the number of batches buffered
    int ask() {
        assert(init_ == true);
        std::lock_guard<std::mutex> lock(mutex_);
        return batch_count_;
    }

    inline bool end_of_file() const { 
        assert(init_ == true);
        return eof_; 
    }

    inline int get_batch_size() const { 
        assert(init_ == true);
        return batch_size_; 
    }

   protected:
    virtual void main() {
        typename io::LineInputFormat::RecordT record;
        eof_ = false;

        while (!eof_) {
            if (batch_num_ == 0) return;

            // Try to fill a batch
            BatchT tmp;
            tmp.reserve(batch_size_);
            for (int i = 0; i < batch_size_; ++i) {
                if (infmt_->next(record)) {
                    tmp.push_back(record.to_string());
                } else {
                    eof_ = true;
                    break;
                }
            }

            {
                // Block if the buffer is full
                std::unique_lock<std::mutex> lock(mutex_);
                while (batch_count_ == batch_num_) {
                    if (batch_num_ == 0) return;
                    load_cv_.wait(lock);
                }
                if (!tmp.empty()) {
                    ++batch_count_;
                    buffer_[end_] = std::move(tmp);
                    if (++end_ >= batch_num_) end_ -= batch_num_;
                }
            }
            get_cv_.notify_one();
        }
        get_cv_.notify_all();
        husky::LOG_I << "loading thread finished";  // for debug
    }

    // input
    std::unique_ptr<io::LineInputFormat> infmt_;
    std::atomic<bool> eof_{false};

    // buffer
    std::vector<BatchT> buffer_;
    int batch_size_;  // the size of each batch
    int batch_num_;  // max buffered batch number
    int batch_count_ = 0;  // unread buffered batch number
    int end_ = 0;  // writer appends to the end_
    int start_ = 0;  // reader reads from the start_
    
    // thread
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable load_cv_;
    std::condition_variable get_cv_;
    bool init_ = false;
};

template <typename T>
class SampleReader {
   public:
    SampleReader() = delete;
    SampleReader(int batch_size, int num_features, AsyncReadBuffer* tbf) : batch_size_(batch_size), num_features_(num_features), tbf_(tbf) {
        assert(tbf->get_batch_size() == batch_size_);  // may not require equality
    }

    int get_batch_size() const {
        return batch_size_;
    }

    virtual std::vector<husky::constants::Key> prepare_next_batch() {
        index_set_.clear();
        typename AsyncReadBuffer::BatchT raw;
        if (!tbf_->get_batch(raw)) {
            this->batch_data_.clear();
            return {0};  // dummy key
        }
        int idx = 0;
        this->batch_data_.resize(raw.size());
        for (auto record : raw) {
            parse_line(record, idx++);
        }
        return {index_set_.begin(), index_set_.end()};
    }

    const std::vector<T>& get_data() {
        return batch_data_;
    }

    inline bool is_empty() { return tbf_->end_of_file() && !tbf_->ask(); }

   protected:
    virtual void parse_line(const std::string& chunk, int pos) = 0;

    AsyncReadBuffer* tbf_;
    int batch_size_;
    int num_features_;
    std::vector<T> batch_data_;
    std::set<husky::constants::Key> index_set_;
};

template <typename FeatureT, typename LabelT, bool is_sparse>
class LIBSVMSampleReader : public SampleReader<husky::lib::ml::LabeledPointHObj<FeatureT, LabelT, is_sparse>> {
   public:
    using Sample = husky::lib::ml::LabeledPointHObj<FeatureT, LabelT, is_sparse>;

    LIBSVMSampleReader(int batch_size, int num_features, AsyncReadBuffer* tbf) : SampleReader<Sample>(batch_size, num_features, tbf) {}
    
    void parse_line(const std::string& chunk, int pos) override {
        if (chunk.empty())
            return;
        boost::char_separator<char> sep(" \t");
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);

        Sample this_obj(this->num_features_);

        bool is_y = true;
        for (auto& w : tok) {
            if (!is_y) {
                boost::char_separator<char> sep2(":");
                boost::tokenizer<boost::char_separator<char>> tok2(w, sep2);
                auto it = tok2.begin();
                int idx = std::stoi(*it++) - 1;  // feature index from 0 to num_fea - 1
                double val = std::stod(*it++);
                this_obj.x.set(idx, val);
                this->index_set_.insert(idx);
            } else {
                this_obj.y = std::stod(w);
                is_y = false;
            }
        }

        this->batch_data_[pos] = std::move(this_obj);
    }
};

template <typename FeatureT, typename LabelT, bool is_sparse>
class TSVSampleReader : public SampleReader<husky::lib::ml::LabeledPointHObj<FeatureT, LabelT, is_sparse>> {
   public:
    using Sample = husky::lib::ml::LabeledPointHObj<FeatureT, LabelT, is_sparse>;

    TSVSampleReader(int batch_size, int num_features, AsyncReadBuffer* tbf) : SampleReader<Sample>(batch_size, num_features, tbf) {}

    void parse_line(const std::string& chunk, int pos) override {
        if (chunk.empty())
            return;
        boost::char_separator<char> sep(" \t");
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);

        Sample this_obj(this->num_features_);

        int i = 0;
        for (auto& w : tok) {
            if (i < this->num_features_) {
                this->index_set_.insert(i);  // TODO: not necessary to store keys for dense form
                this_obj.x.set(i++, std::stod(w));
            } else {
                this_obj.y = std::stod(w);
            }
        }

        this->batch_data_[pos] = std::move(this_obj);
    }
};

}  // namespace anonymous
}  // husky
