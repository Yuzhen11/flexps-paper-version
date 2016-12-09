#pragma once

#include <vector>
#include <unordered_map>

#include "husky/core/zmq_helpers.hpp"
#include "husky/base/log.hpp"

namespace husky {

class MasterConnection {
public:
    MasterConnection() = delete;
    MasterConnection(zmq::context_t& context_, const std::string& bind_addr)
        :context(context_), 
        recv_socket(context, ZMQ_PULL) {
        recv_socket.bind(bind_addr);
        base::log_msg("[MasterConnection]: Bind to "+bind_addr);
    }

    void add_proc(int proc_id, const std::string& remote_addr) {
        zmq::socket_t socket(context, ZMQ_PUSH);
        socket.connect(remote_addr);
        base::log_msg("[MasterConnection]: Connect to "+remote_addr);
        proc_sockets.emplace(proc_id, std::move(socket));
    }
    auto& get_send_sockets() {
        return proc_sockets;
    }

    auto& get_recv_socket() {
        return recv_socket;
    }

private:
    zmq::context_t& context;
    std::unordered_map<int, zmq::socket_t> proc_sockets;  // send tasks to proc {proc_id, socket}
    zmq::socket_t recv_socket;  // recv info from proc 
};

}  // namespace husky
