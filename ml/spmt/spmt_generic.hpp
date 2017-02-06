#pragma once

#include <chrono>
#include <mutex>
#include <condition_variable>

#include "core/info.hpp"
#include "husky/base/exception.hpp"
#include "husky/base/serialization.hpp"
#include "husky/core/zmq_helpers.hpp"

#include "ml/common/mlworker.hpp"
#include "ml/spmt/consistency_controller.hpp"
#include "ml/spmt/ssp_consistency_controller.hpp"
#include "ml/spmt/bsp_consistency_controller.hpp"

#include "kvstore/kvstore.hpp"

#include "core/color.hpp"

namespace ml {
namespace spmt {

struct Model {
    std::mutex mtx;
    std::vector<float> params;
};

class SPMTGenericWorker : public common::GenericMLWorker {
   public:
    SPMTGenericWorker() = delete;

    /*
     * @param type: 0 for ssp, 1 for bsp
     */
    template <typename... Args>
    SPMTGenericWorker(int model_id, zmq::context_t& context, const husky::Info& info, const std::string& type, Args&&... args)
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
            if (type == "BSP") {
                p_controller_  = new BSPConsistencyController;
            } else if (type == "SSP") {
                p_controller_ = new SSPConsistencyController;
            } else {
                assert(false);
            }
            p_controller_->Init(info.get_num_local_workers());

            model_ = new Model;
            model_->params.resize(std::forward<Args>(args)...);
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
            p_controller_ = reinterpret_cast<AbstractConsistencyController*>(ptr);
            model_ = reinterpret_cast<Model*>(ptr2);
        }
    }

    virtual void Load() override {}
    virtual void Dump() override {}

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        p_controller_->BeforePush(info_.get_cluster_id());
        {
            // Acquire lock
            std::lock_guard<std::mutex> lck(model_->mtx);
            // Directly update the model
            assert(keys.size() == vals.size());
            for (size_t i = 0; i < keys.size(); i++) {
                assert(i < model_->params.size());
                model_->params[keys[i]] += vals[i];
            }
        }
        p_controller_->AfterPush(info_.get_cluster_id());
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals) override {
        p_controller_->BeforePull(info_.get_cluster_id());
        {
            // Acquire lock
            std::lock_guard<std::mutex> lck(model_->mtx);
            // Directly access the model
            vals->resize(keys.size());
            for (size_t i = 0; i < keys.size(); i++) {
                assert(keys[i] < model_->params.size());
                (*vals)[i] = model_->params[keys[i]];
            }
        }
        p_controller_->AfterPull(info_.get_cluster_id());
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
    Model* model_ = nullptr;

    const husky::Info& info_;
    zmq::context_t& context_;
    zmq::socket_t socket_;
    int model_id_;

    AbstractConsistencyController* p_controller_;
};

}  // namespace spmt
}  // namespace ml
