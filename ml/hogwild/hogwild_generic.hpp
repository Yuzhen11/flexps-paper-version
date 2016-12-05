#pragma once

#include "base/serialization.hpp"
#include "base/exception.hpp"
#include "core/zmq_helpers.hpp"
#include "core/info.hpp"

#include "ml/common/mlworker.hpp"

namespace ml {
namespace hogwild {

/*
 * For the HogwildGenericModel, the type ModelType is now fixed to std::vector<float>
 */
class HogwildGenericModel : public common::GenericMLWorker {
public:
    HogwildGenericModel() = delete;
    /*
     * constructor to construct a hogwild model
     * \param context zmq_context
     * \param info info in this instance
     * \param args variable args to initialize the variables
     */
    template<typename... Args>
    HogwildGenericModel(zmq::context_t& context, husky::Info& info, Args&&... args)
        : info_(info), 
          context_(context),
          socket_(context, info.cluster_id == 0 ? ZMQ_ROUTER:ZMQ_REQ) {
        int task_id = info_.task->get_id();
        // check valid
        if (!isValid()) {
            throw husky::base::HuskyException("[Hogwild] threads are not in the same machine. Task is:"+std::to_string(task_id));
        }
        // bind and connect
        if (info_.cluster_id == 0) {  // leader
            socket_.bind("inproc://tmp-"+std::to_string(task_id));
        } else {
            socket_.connect("inproc://tmp-"+std::to_string(task_id));
        }

        if (info_.cluster_id == 0) {
            // use args to initialize the variable
            model = new std::vector<float>(std::forward<Args>(args)...);
        }

        // TODO may not be portable, pointer size problem
        if (info_.cluster_id == 0) {  // leader
            std::vector<std::string> identity_store;
            auto ptr = reinterpret_cast<std::uintptr_t>(model);
            for (int i = 0; i < info_.num_local_threads-1; ++ i) {
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
            model = reinterpret_cast<std::vector<float>*>(ptr);
            husky::base::log_msg(std::to_string(model->size()));
        }
    }

    /*
     * destructor
     * 1. sync() and 2. leader delete the model
     */
    ~HogwildGenericModel() {
        sync();
        if (info_.cluster_id == 0) {
            delete model;
        }
    }

    /*
     * Put/Get APIs
     */
    virtual void Put(int key, float val) {
        (*model)[key] = val;
    }
    virtual float Get(int key) {
        return (*model)[key];
    }

    /*
     * Get the model
     */
    std::vector<float>* get() {
        return model;
    }

    /*
     * Serve as a barrier
     */
    void sync() {
        if (info_.cluster_id == 0) {  // leader
            std::vector<std::string> identity_store;
            for (int i = 0; i < info_.num_local_threads-1; ++ i) {
                std::string s = husky::zmq_recv_string(&socket_);
                identity_store.push_back(std::move(s));
                husky::zmq_recv_dummy(&socket_);  // delimiter
                husky::zmq_recv_dummy(&socket_);  // dummy msg
            }
            // collect all and then reply
            for (auto& identity : identity_store) {
                husky::zmq_sendmore_string(&socket_, identity);
                husky::zmq_sendmore_dummy(&socket_);  // delimiter
                husky::zmq_send_dummy(&socket_);  // dummy msg
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
        return info_.num_local_threads == info_.num_global_threads;
    }

    husky::Info& info_;
    zmq::context_t& context_;
    zmq::socket_t socket_;

    // pointer to the real model
    std::vector<float>* model = nullptr;
};

}  // namespace hogwild
}  // namespace ml
