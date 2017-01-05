#pragma once

#include "husky/base/log.hpp"

namespace husky {

// This class is use to connect to cluster_manager
class ClusterManagerConnector {
   public:
    ClusterManagerConnector() = delete;
    ClusterManagerConnector(zmq::context_t* context_, const std::string& bind_addr, const std::string& cluster_manager_addr,
                    const std::string& host_name)
        : context_(context_),
          recv_socket_(*context_, ZMQ_PULL),
          send_socket_(*context_, ZMQ_PUSH),
          bind_addr_(bind_addr),
          cluster_manager_addr_(cluster_manager_addr),
          host_name_(host_name) {
        recv_socket_.bind(bind_addr_);
        husky::LOG_I << "[ClusterManagerConnector]: Bind to " + bind_addr_;
        send_socket_.connect(cluster_manager_addr_);
        husky::LOG_I << "[ClusterManagerConnector]: Connect to " + cluster_manager_addr_;

        local_addr_ = bind_addr_;
        auto pos = local_addr_.find("*");
        local_addr_.replace(pos, 1, host_name_);
        husky::LOG_I << "[ClusterManagerConnector]: Local address: " + local_addr_;
    }

    ClusterManagerConnector(const ClusterManagerConnector&) = delete;
    ClusterManagerConnector& operator=(const ClusterManagerConnector&) = delete;

    ClusterManagerConnector(ClusterManagerConnector&& rhs) = default;
    ClusterManagerConnector& operator=(ClusterManagerConnector&& rhs) = default;

    auto& get_recv_socket() { return recv_socket_; }
    auto& get_send_socket() { return send_socket_; }
    zmq::context_t& get_context() { return *context_; }
    // Newly spawned threads need to get a socket to connect to the main loop
    zmq::socket_t get_socket_to_recv() {
        zmq::socket_t socket(*context_, ZMQ_PUSH);
        socket.connect(local_addr_);
        return socket;
    }

   private:
    std::string host_name_;
    std::string bind_addr_;
    std::string cluster_manager_addr_;
    std::string local_addr_;
    zmq::context_t* context_;
    zmq::socket_t recv_socket_;
    zmq::socket_t send_socket_;
};

}  // namespace husky
