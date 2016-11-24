#pragma once

#include <string>
#include "zmq.hpp"
#include "core/config.hpp"
#include "core/worker_info.hpp"
#include "core/mailbox.hpp"

namespace husky {

struct Global {
    Config config;
    zmq::context_t* zmq_context_ptr = nullptr;
    WorkerInfo worker_info;
    std::vector<LocalMailbox*> mailboxes;
    LocalMailbox* kv_mailbox;
};

class Context {
public:
    static void init_global() {
        global = new Global();
        global->zmq_context_ptr = new zmq::context_t();
    }
    static void finalize_global() {
        auto p = global->zmq_context_ptr;
        delete global;
        delete p;
        global = nullptr;
    }

    static Global* get_global()  { return global; }
    static Config* get_config() { return &(global->config); }
    static std::string get_param(const std::string& key) { return global->config.get_param(key); }
    static auto& get_params() { return global->config.get_params(); }
    static zmq::context_t& get_zmq_context() { return *(global->zmq_context_ptr); }
    static WorkerInfo* get_worker_info() { return &(global->worker_info); }
    static void set_mailboxes(const std::vector<LocalMailbox*>& mailboxes_) {
        global->mailboxes = mailboxes_;
    }
    static void set_kv_mailbox(LocalMailbox* mailbox) {
        global->kv_mailbox = mailbox;
    }
    static LocalMailbox* get_mailbox(int id) {
        return global->mailboxes[id];
    }
    static LocalMailbox* get_kv_mailbox() {
        return global->kv_mailbox;
    }
    static std::string get_recver_bind_addr() { return "tcp://*:" + std::to_string(global->config.get_comm_port()); }

private:
    static Global* global;
};

}  // namespace husky
