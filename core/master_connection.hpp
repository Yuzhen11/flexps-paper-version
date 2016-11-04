#pragma once

#include <vector>
#include <unordered_map>

#include "core/zmq_helpers.hpp"
#include "base/debug.hpp"

#include "zmq.hpp"

namespace husky {

class MasterConnection {
public:
    MasterConnection() = delete;
    MasterConnection(zmq::context_t& context_):context(context_) {}

    void add_proc(int proc_id, const std::string& remote_addr) {
        zmq::socket_t socket(context, ZMQ_PUSH);
        socket.connect(remote_addr);
        std::cout << "[MasterConnection]: Successfully connect to "+remote_addr << std::endl;
        proc_sockets.emplace(proc_id, std::move(socket));
    }
    auto& get_sockets() {
        return proc_sockets;
    }

private:
    zmq::context_t& context;
    std::unordered_map<int, zmq::socket_t> proc_sockets;
};

}  // namespace husky
