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
#include "ml/spmt/asp_consistency_controller.hpp"
#include "ml/spmt/ssp_consistency_controller.hpp"
#include "ml/spmt/bsp_consistency_controller.hpp"
#include "ml/model/integral_model.hpp"
#include "ml/model/chunk_based_mt_model.hpp"

#include "kvstore/kvstore.hpp"

#include "core/color.hpp"

namespace ml {
namespace spmt {

class SPMTGenericWorker : public common::GenericMLWorker {
   public:
    SPMTGenericWorker() = delete;

    /*
     * @param type: 0 for ssp, 1 for bsp
     */
    template <typename... Args>
    SPMTGenericWorker(int model_id, zmq::context_t& context, const husky::Info& info, const std::string& type, Args&&... args)
        : info_(info),
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
            } else if (type == "ASP") {
                p_controller_ = new ASPConsistencyController;
            } else {
                assert(false);
            }
            p_controller_->Init(info.get_num_local_workers());

            model_ = (model::Model*) new model::ChunkBasedMTLockModel(model_id, std::forward<Args>(args)...);
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
            model_ = reinterpret_cast<model::Model*>(ptr2);
        }
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        p_controller_->BeforePush(info_.get_cluster_id());
        model_->Push(keys, vals);
        p_controller_->AfterPush(info_.get_cluster_id());
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals) override {
        p_controller_->BeforePull(info_.get_cluster_id());
        model_->Pull(keys, vals, info_.get_local_id());
        p_controller_->AfterPull(info_.get_cluster_id());
    }

    // For v2
    // TODO: Now, the v2 APIs for spmt still need copy,
    // Later, we may use brunching to facilitate zero-copy when doing single/hogwild
    virtual void Prepare_v2(const std::vector<husky::constants::Key>& keys) override {
        keys_ = const_cast<std::vector<husky::constants::Key>*>(&keys);
        Pull(keys, &vals_);
        delta_.clear();
        delta_.resize(keys.size());
    }
    virtual float Get_v2(husky::constants::Key idx) override { return vals_[idx]; }
    virtual void Update_v2(husky::constants::Key idx, float val) override {
        delta_[idx] += val;
        vals_[idx] += val;
    }
    virtual void Update_v2(const std::vector<float>& vals) override {
        assert(vals.size() == vals_.size());
        for (size_t i = 0; i < vals.size(); ++i) {
            vals_[i] += vals[i];
            delta_[i] += vals[i];
        }
    }
    virtual void Clock_v2() override { Push(*keys_, delta_); }

    virtual void Load() override {
        if (info_.get_cluster_id() == 0) {
            model_->Load(info_.get_local_id(), "");
        }
        Sync();
    }

    virtual void Dump() override {
        Sync();
        if (info_.get_cluster_id() == 0) {
            model_->Dump(info_.get_local_id(), "");
        }
        Sync();
    }

    /*
     * Serve as a barrier
     */
    virtual void Sync() override {
        if (info_.get_cluster_id() == 0) {  // leader
            std::vector<std::string> identity_store;
            for (int i = 0; i < info_.get_num_local_workers() - 1; ++i) {
                std::string s = husky::zmq_recv_string(&socket_);
                identity_store.push_back(std::move(s));
                husky::zmq_recv_dummy(&socket_);  // delimiter
                husky::zmq_recv_dummy(&socket_);  // dummy msg
            }
            // collect all and then reply
            for (auto& identity : identity_store) {
                husky::zmq_sendmore_string(&socket_, identity);
                husky::zmq_sendmore_dummy(&socket_);  // delimiter
                husky::zmq_send_dummy(&socket_);      // dummy msg
            }
        } else {
            husky::zmq_send_dummy(&socket_);
            husky::zmq_recv_dummy(&socket_);
        }
    }

   private:
    /*
     * check whether all the threads are in the same machine
     */
    bool isValid() {
        return info_.get_num_local_workers() == info_.get_num_workers();
    }

    // pointer to the real model
    model::Model* model_ = nullptr;

    const husky::Info& info_;
    zmq::context_t& context_;
    zmq::socket_t socket_;

    AbstractConsistencyController* p_controller_;

    // For v2
    // Pointer to keys
    std::vector<husky::constants::Key>* keys_;
    std::vector<float> vals_;
    std::vector<float> delta_;
};

}  // namespace spmt
}  // namespace ml