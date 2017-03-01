#pragma once

#include <unordered_map>
#include <vector>

#include "husky/base/log.hpp"
#include "husky/core/zmq_helpers.hpp"

namespace husky {

class ClusterManagerConnection {
   public:
    ClusterManagerConnection() = delete;
    ClusterManagerConnection(zmq::context_t* context, const std::string& bind_addr)
        : bind_addr_(bind_addr), context_(context), recv_socket_(*context_, ZMQ_PULL) {
        recv_socket_.bind(bind_addr);
        husky::LOG_I << "[ClusterManagerConnection]: Bind to " + bind_addr;
    }

    ClusterManagerConnection(const ClusterManagerConnection&) = delete;
    ClusterManagerConnection& operator=(const ClusterManagerConnection&) = delete;

    ClusterManagerConnection(ClusterManagerConnection&& rhs) = default;
    ClusterManagerConnection& operator=(ClusterManagerConnection&& rhs) = default;

    void add_proc(int proc_id, const std::string& remote_addr) {
        zmq::socket_t socket(*context_, ZMQ_PUSH);
        socket.connect(remote_addr);
        husky::LOG_I << "[ClusterManagerConnection]: Connect to " + remote_addr;
        proc_sockets_.emplace(proc_id, std::move(socket));
    }

    std::unordered_map<int, zmq::socket_t>& get_send_sockets() { return proc_sockets_; }

    zmq::socket_t& get_send_socket(int id) { 
        assert(proc_sockets_.find(id) != proc_sockets_.end());
        return proc_sockets_.find(id)->second;
    }

    zmq::context_t* get_context() { return context_; }

    std::string get_cluster_manager_addr() {
        return bind_addr_;
    }

    zmq::socket_t& get_recv_socket() { return recv_socket_; }

   private:
    zmq::context_t* context_;
    std::unordered_map<int, zmq::socket_t> proc_sockets_;  // send tasks to proc {proc_id, socket}
    zmq::socket_t recv_socket_;                            // recv info from proc
    std::string bind_addr_;
};

}  // namespace husky
