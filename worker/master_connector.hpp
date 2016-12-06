#pragma once

#include "base/log.hpp"

namespace husky {

// This class is use to connect to master
class MasterConnector {
public:
    MasterConnector() = delete;
    MasterConnector(zmq::context_t& context_, const std::string& bind_addr_, const std::string& master_addr_, const std::string& host_name_)
        :context(context_),
        recv_socket(context, ZMQ_PULL),
        send_socket(context, ZMQ_PUSH),
        bind_addr(bind_addr_),
        master_addr(master_addr_),
        host_name(host_name_) {
        recv_socket.bind(bind_addr);
        base::log_msg("[MasterConnector]: Bind to "+bind_addr);
        send_socket.connect(master_addr);
        base::log_msg("[MasterConnector]: Connect to "+master_addr);

        local_addr = bind_addr;
        auto pos = local_addr.find("*");
        local_addr.replace(pos, 1, host_name);
        base::log_msg("[MasterConnector]: Local address: "+local_addr);
    }
    auto& get_recv_socket() {
        return recv_socket;
    }
    auto& get_send_socket() {
        return send_socket;
    }
    zmq::context_t& get_context() {
        return context;
    }
    // Newly spawned threads need to get a socket to connect to the main loop
    zmq::socket_t get_socket_to_recv() {
        zmq::socket_t socket(context, ZMQ_PUSH);
        socket.connect(local_addr);
        return socket;
    }

private:
    std::string host_name;
    std::string bind_addr;
    std::string master_addr;
    std::string local_addr;
    zmq::context_t& context;
    zmq::socket_t recv_socket;
    zmq::socket_t send_socket;

};

}  // namespace husky
