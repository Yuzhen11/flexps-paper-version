#pragma once

#include <string.h>
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

template <typename Sample>
class AsyncReadParseBuffer {
   public:
    struct BatchT {
        BatchT() = default;
        explicit BatchT(int batch_size) { data.reserve(batch_size); }

        std::vector<Sample> data;
        std::set<husky::constants::Key> keys;
    };

    AsyncReadParseBuffer() = default;

    AsyncReadParseBuffer(const AsyncReadParseBuffer&) = delete;
    AsyncReadParseBuffer& operator=(const AsyncReadParseBuffer&) = delete;
    AsyncReadParseBuffer(AsyncReadParseBuffer&&) = delete;
    AsyncReadParseBuffer& operator=(AsyncReadParseBuffer&&) = delete;

    // destructor: stop thread and clear buffer
    ~AsyncReadParseBuffer() {
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
     * \param batch_num the number of batches
     * \param num_features the number of features in a sample
     */
    void init(const std::string& url, int task_id, int num_threads, int batch_size, int batch_num, int num_features) {
        if (init_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        if (init_) return;

        // The initialization work
        batch_size_ = batch_size;
        batch_num_ = batch_num;
        num_features_ = num_features;
        buffer_.resize(batch_num);
        infmt_.reset(new io::LineInputFormatML(num_threads, task_id));
        infmt_->set_input(url);
        thread_ = std::thread(&AsyncReadParseBuffer::main, this);
        init_ = true;
    }

    // store batch_size_ samples in the batch and return true if success
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
            BatchT tmp(batch_size_);
            for (int i = 0; i < batch_size_; ++i) {
                if (infmt_->next(record)) {
                    parse_line(record, tmp);
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
                if (!tmp.data.empty()) {
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

    virtual void parse_line(const boost::string_ref& chunk, BatchT& batch) = 0;

    // input
    std::unique_ptr<io::LineInputFormatML> infmt_;
    std::atomic<bool> eof_{false};
    int num_features_ = 0;

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

template <typename Sample>
class SimpleSampleReader{
   public:
    SimpleSampleReader() = delete;
    SimpleSampleReader(AsyncReadParseBuffer<Sample>* tbf) : tbf_(tbf) {}

    virtual std::vector<husky::constants::Key> prepare_next_batch() {
        typename AsyncReadParseBuffer<Sample>::BatchT batch;
        if (!tbf_->get_batch(batch)) {
            batch_data_.clear();
            return {0};  // dummy key
        }
        batch_data_ = std::move(batch.data);
        return {batch.keys.begin(), batch.keys.end()};
    }

    const std::vector<Sample>& get_data() {
        return batch_data_;
    }

    inline bool is_empty() { return tbf_->end_of_file() && !tbf_->ask(); }
    inline int get_batch_size() const { return tbf_->get_batch_size(); }

   protected:
    AsyncReadParseBuffer<Sample>* tbf_;
    std::vector<Sample> batch_data_;
};

template <typename Sample>
class LIBSVMAsyncReadParseBuffer : public AsyncReadParseBuffer<Sample> {
   public:
    LIBSVMAsyncReadParseBuffer() : AsyncReadParseBuffer<Sample>() {}
    
    void parse_line(const boost::string_ref& chunk, typename AsyncReadParseBuffer<Sample>::BatchT& batch) override {
        if (chunk.empty()) return;

        Sample this_obj(this->num_features_);

        // parse
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
                batch.keys.insert(idx);
                i = 1;
            } else if (i == 1) {
                val = std::atol(tok);
                this_obj.x.set(idx, val);
                i = 0;
            } else {
                this_obj.y = std::atol(tok);
                i = 0;
            }
            // Next key/value pair
            tok = strtok_r(NULL, " \t:", &pos);
        }
        batch.data.push_back(std::move(this_obj));
    }
};

template <typename Sample>
class TSVAsyncReadParseBuffer : public AsyncReadParseBuffer<Sample> {
   public:
    TSVAsyncReadParseBuffer() : AsyncReadParseBuffer<Sample>() {}

    void parse_line(const boost::string_ref& chunk, typename AsyncReadParseBuffer<Sample>::BatchT& batch) override {
        if (chunk.empty()) return;

        Sample this_obj(this->num_features_);

        char* pos;
        std::unique_ptr<char> chunk_ptr(new char[chunk.size() + 1]);
        strncpy(chunk_ptr.get(), chunk.data(), chunk.size());
        chunk_ptr.get()[chunk.size()] = '\0';
        char* tok = strtok_r(chunk_ptr.get(), " \t", &pos);

        int i = 0;
        while (tok != NULL) {
            if (i < this->num_features_) {
                batch.keys.insert(i);  // TODO: not necessary to store keys for dense form
                this_obj.x.set(i++, std::stol(tok));
            } else {
                this_obj.y = std::stol(tok);
            }
            // Next key/value pair
            tok = strtok_r(NULL, " \t", &pos);
        }
        batch.data.push_back(std::move(this_obj));
    }
};

}  // namespace anonymous
}  // husky
