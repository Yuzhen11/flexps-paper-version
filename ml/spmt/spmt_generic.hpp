#pragma once

#include <chrono>
#include <mutex>
#include <condition_variable>

#include "core/info.hpp"
#include "husky/base/exception.hpp"
#include "husky/base/serialization.hpp"
#include "husky/core/zmq_helpers.hpp"

#include "ml/common/mlworker.hpp"

#include "kvstore/kvstore.hpp"

#include "core/color.hpp"

namespace ml {
namespace spmt {

struct Controller {
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<int> worker_progress;
    std::vector<int> clock_count;
    int staleness = 1;
    int min_clock = 0;
};

class SPMTGenericWorker : public common::GenericMLWorker {
   public:
    SPMTGenericWorker() = delete;

    template <typename... Args>
    SPMTGenericWorker(int model_id, zmq::context_t& context, const husky::Info& info, Args&&... args)
        : info_(info),
          model_id_(model_id),
          context_(context),
          socket_(context, info.get_cluster_id() == 0 ? ZMQ_ROUTER : ZMQ_REQ) {
        int task_id = info_.get_task()->get_id();
        // check valid
        if (!isValid()) {
            throw husky::base::HuskyException("[Hogwild] threads are not in the same machine. Task is:" +
                                              std::to_string(task_id));
        }
        // bind and connect
        if (info_.get_cluster_id() == 0) {  // leader
            socket_.bind("inproc://tmp-" + std::to_string(task_id));
        } else {
            socket_.connect("inproc://tmp-" + std::to_string(task_id));
        }

        if (info_.get_cluster_id() == 0) {
            p_controller_ = new Controller;
            p_controller_->worker_progress.resize(info_.get_num_local_workers(), 0);
            model_ = new std::vector<float>(std::forward<Args>(args)...);
        }
        if (info_.get_cluster_id() == 0) {
            std::vector<std::string> identity_store;
            auto ptr = reinterpret_cast<std::uintptr_t>(p_controller_);
            auto ptr2 = reinterpret_cast<std::uintptr_t>(model_);
            for (int i = 0; i < info_.get_num_local_workers() - 1; ++i) {
                std::string s = husky::zmq_recv_string(&socket_);
                identity_store.push_back(std::move(s));
                husky::zmq_recv_dummy(&socket_);  // delimiter
                husky::zmq_recv_int32(&socket_);
            }
            // collect all and then reply
            for (auto& identity : identity_store) {
                husky::zmq_sendmore_string(&socket_, identity);
                husky::zmq_sendmore_dummy(&socket_);  // delimiter
                husky::zmq_sendmore_int64(&socket_, ptr);
                husky::zmq_send_int64(&socket_, ptr2);
            }
        } else {
            husky::zmq_send_int32(&socket_, int());
            auto ptr = husky::zmq_recv_int64(&socket_);
            auto ptr2 = husky::zmq_recv_int64(&socket_);
            p_controller_ = reinterpret_cast<Controller*>(ptr);
            model_ = reinterpret_cast<std::vector<float>*>(ptr2);
        }
    }

    virtual void Load() override {}
    virtual void Dump() override {}

    Controller& get_controller() {
        return *p_controller_;
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        // Acquire lock
        std::unique_lock<std::mutex> lck(p_controller_->mtx);

        // Directly update the model
        assert(keys.size() == vals.size());
        for (size_t i = 0; i < keys.size(); i++) {
            assert(i < model_->size());
            (*model_)[keys[i]] += vals[i];
        }

        int src = info_.get_cluster_id();
        if (src > p_controller_->worker_progress.size())
            p_controller_->worker_progress.resize(src + 1);
        int progress = p_controller_->worker_progress[src];
        if (progress >= p_controller_->clock_count.size())
            p_controller_->clock_count.resize(progress + 1);
        p_controller_->clock_count[progress] += 1;
        if (progress == p_controller_->min_clock && p_controller_->clock_count[p_controller_->min_clock] == info_.get_num_workers()) {
            p_controller_->min_clock += 1;
            // release all pull blocked at min_clock
            p_controller_->cv.notify_all();
        }
        p_controller_->worker_progress[src] += 1;
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals) override {
        // Acquire lock
        std::unique_lock<std::mutex> lck(p_controller_->mtx);

        int src = info_.get_cluster_id();
        if (src >= p_controller_->worker_progress.size())
            p_controller_->worker_progress.resize(src + 1);
        int expected_min_lock = p_controller_->worker_progress[src] - p_controller_->staleness;
        while (expected_min_lock > p_controller_->min_clock) {
            p_controller_->cv.wait(lck);
        }

        // Directly access the model
        vals->resize(keys.size());
        for (size_t i = 0; i < keys.size(); i++) {
            assert(keys[i] < model_->size());
            (*vals)[i] = (*model_)[keys[i]];
        }
    }

   private:
    /*
     * check whether all the threads are in the same machine
     */
    bool isValid() {
        // husky::base::log_msg("locals: " + std::to_string(info_.get_num_local_workers()) + " globals:" +
        //                      std::to_string(info_.get_num_workers()));
        return info_.get_num_local_workers() == info_.get_num_workers();
    }

    // pointer to the real model
    std::vector<float>* model_ = nullptr;

    const husky::Info& info_;
    zmq::context_t& context_;
    zmq::socket_t socket_;
    int model_id_;

    Controller* p_controller_;
};

}  // namespace spmt
}  // namespace ml
