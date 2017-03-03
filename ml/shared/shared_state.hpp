#pragma once

#include "husky/base/exception.hpp"
#include "husky/core/zmq_helpers.hpp"

namespace ml {

/*
 * A class to store the shared_state
 *
 * One task can only have one SharedState
 *
 * Only leader call Init and SyncState, then others can 
 * use Get() method to get the pointer to the shared_state
 *
 * User is in charge of take care of the shared_state memroy
 */
template <typename T>
class SharedState {
   public:
    SharedState() = delete;
    SharedState(const SharedState&) = delete;
    SharedState& operator=(const SharedState&) = delete;
    SharedState(SharedState&&) = delete;
    SharedState& operator=(SharedState&&) = delete;

    SharedState(int task_id, bool is_leader, int num_threads, zmq::context_t& context)
        : task_id_(task_id),
          is_leader_(is_leader),
          num_threads_(num_threads),
          context_(context),
          socket_(context, is_leader == true ? ZMQ_ROUTER : ZMQ_REQ) {
        // bind and connect
        if (is_leader_ == true) {  // leader
            socket_.bind("inproc://tmp-" + std::to_string(task_id));
        } else {
            socket_.connect("inproc://tmp-" + std::to_string(task_id));
        }
    }

    /*
     * Only leader call Init, and the SharedState will take the ownership of the newed variable
     */
    void Init(T* shared) {
        assert(is_leader_ == true);
        shared_ = shared;
    }

    /*
     * Sync the state
     */
    void SyncState() {
        if (is_leader_ == true) {
            assert(shared_ != nullptr);
            std::vector<std::string> identity_store;
            auto ptr = reinterpret_cast<std::uintptr_t>(shared_);
            // 1. Collect all
            for (int i = 0; i < num_threads_ - 1; ++i) {
                std::string s = husky::zmq_recv_string(&socket_);
                identity_store.push_back(std::move(s));
                husky::zmq_recv_dummy(&socket_);  // delimiter
                husky::zmq_recv_int32(&socket_);
            }
            // 2. Send to all
            for (auto& identity : identity_store) {
                husky::zmq_sendmore_string(&socket_, identity);
                husky::zmq_sendmore_dummy(&socket_);  // delimiter
                husky::zmq_send_int64(&socket_, ptr);
            }
        } else {
            // 1. Send to 0
            husky::zmq_send_int32(&socket_, int());
            // 2. Receive from 0
            auto ptr = husky::zmq_recv_int64(&socket_);
            shared_ = reinterpret_cast<T*>(ptr);
        }
    }

    /*
     * Get the shared state
     */
    T* Get() {
        return shared_;
    }
    const T* Get() const {
        return shared_;
    }

    /*
     * Process level Barrier
     */
    void Barrier() {
        if (is_leader_ == true) {  // leader
            std::vector<std::string> identity_store;
            for (int i = 0; i < num_threads_ - 1; ++i) {
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
    // the shared state
    T* shared_ = nullptr;

    // utils
    int task_id_;
    bool is_leader_;
    int num_threads_;
    zmq::context_t& context_;
    zmq::socket_t socket_;
};

}  // namespace ml

