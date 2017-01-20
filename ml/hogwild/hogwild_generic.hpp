#pragma once

#include <chrono>

#include "core/info.hpp"
#include "husky/base/exception.hpp"
#include "husky/base/serialization.hpp"
#include "husky/core/zmq_helpers.hpp"

#include "ml/common/mlworker.hpp"

#include "kvstore/kvstore.hpp"

#include "core/color.hpp"

namespace ml {
namespace hogwild {

/*
 * For the HogwildGenericWorker, the type ModelType is now fixed to std::vector<float>
 */
class HogwildGenericWorker : public common::GenericMLWorker {
   public:
    HogwildGenericWorker() = delete;
    /*
     * constructor to construct a hogwild model
     * \param context zmq_context
     * \param info info in this instance
     * \param args variable args to initialize the variables
     */
    template <typename... Args>
    HogwildGenericWorker(int model_id, zmq::context_t& context, const husky::Info& info, Args&&... args)
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
            // use args to initialize the variable
            model_ = new std::vector<float>(std::forward<Args>(args)...);
        }

        // TODO may not be portable, pointer size problem
        if (info_.get_cluster_id() == 0) {  // leader
            std::vector<std::string> identity_store;
            auto ptr = reinterpret_cast<std::uintptr_t>(model_);
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
                husky::zmq_send_int64(&socket_, ptr);
            }
        } else {
            husky::zmq_send_int32(&socket_, int());
            auto ptr = husky::zmq_recv_int64(&socket_);
            model_ = reinterpret_cast<std::vector<float>*>(ptr);
        }
    }

    /*
     * destructor
     * 1. Sync() and 2. leader delete the model
     */
    ~HogwildGenericWorker() {
        Sync();
        if (info_.get_cluster_id() == 0) {
            delete model_;
        }
    }

    void print_model() const {
        // debug
        for (size_t i = 0; i < model_->size(); ++i)
            husky::LOG_I << std::to_string((*model_)[i]);
    }

    /*
     * Get parameters from global kvstore
     */
    virtual void Load() override {
        if (info_.get_cluster_id() == 0) {
            husky::LOG_I << "[Hogwild] loading model_id:" + std::to_string(model_id_) + " local_id:"+
                             std::to_string(info_.get_local_id()) + "model_size: " + std::to_string(model_->size());
            auto start_time = std::chrono::steady_clock::now();
            auto* kvworker = kvstore::KVStore::Get().get_kvworker(info_.get_local_id());
            std::vector<husky::constants::Key> keys(model_->size());
            for (size_t i = 0; i < keys.size(); ++i)
                keys[i] = i;
            int ts = kvworker->Pull(model_id_, keys, model_);
            kvworker->Wait(model_id_, ts);
            auto end_time = std::chrono::steady_clock::now();
            husky::LOG_I << "[Hogwild] Load done and Load time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count() << " ms";
            // print_model();
        }
        Sync();
    }
    /*
     * Put the parameters to global kvstore
     */
    virtual void Dump() override {
        Sync();
        if (info_.get_cluster_id() == 0) {
            // husky::LOG_I << "[Hogwild] dumping";

            auto* kvworker = kvstore::KVStore::Get().get_kvworker(info_.get_local_id());

            std::vector<husky::constants::Key> keys(model_->size());
            for (size_t i = 0; i < keys.size(); ++i)
                keys[i] = i;
            int ts = kvworker->Push(model_id_, keys, *model_);
            kvworker->Wait(model_id_, ts);
        }
        Sync();
    }

    /*
     * Put/Get Push/Pull APIs
     */
    virtual void Put(size_t key, float val) {
        assert(key < model_->size());
        (*model_)[key] += val;
    }
    virtual float Get(size_t key) {
        assert(key < model_->size());
        return (*model_)[key];
    }
    virtual void Push(const std::vector<husky::constants::Key>&keys, const std::vector<float>& vals) override {
        assert(keys.size() == vals.size());
        for (size_t i = 0; i < keys.size(); i++) {
            assert(i < model_->size());
            (*model_)[keys[i]] += vals[i];
        }
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals) override {
        vals->resize(keys.size());
        for (size_t i = 0; i < keys.size(); i++) {
            assert(i < model_->size());
            (*vals)[i] = (*model_)[keys[i]];
        }
    }
    

    /*
     * Get the model
     */
    std::vector<float>* get() { return model_; }

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


    // For v2
    virtual void Prepare_v2(std::vector<husky::constants::Key>& keys) override {
        keys_ = &keys;
    }
    virtual float Get_v2(size_t idx) override {
        return (*model_)[(*keys_)[idx]];
    }
    virtual void Update_v2(size_t idx, float val) override {
        (*model_)[(*keys_)[idx]] += val;
    }
    virtual void Update_v2(const std::vector<float>& vals) override {
        assert(vals.size() == keys_->size());
        for (size_t i = 0; i < keys_->size(); ++ i) {
            assert((*keys_)[i] < model_->size());
            (*model_)[(*keys_)[i]] += vals[i];
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

    const husky::Info& info_;
    zmq::context_t& context_;
    zmq::socket_t socket_;

    // pointer to the real model
    std::vector<float>* model_ = nullptr;

    int model_id_;

    // For v2
    // Pointer to keys
    std::vector<husky::constants::Key>* keys_;
};

}  // namespace hogwild
}  // namespace ml
