#pragma once

#include "base/debug.hpp"

#include "zmq.hpp" 
#include "base/serialization.hpp"
#include "core/constants.hpp"
#include "core/instance.hpp"
#include "core/instance_manager.hpp"
#include "core/worker_info.hpp"
#include "core/zmq_helpers.hpp"

namespace husky {

class MainLoop {
public:
    MainLoop() = delete;
    MainLoop(WorkerInfo&& worker_info_, zmq::context_t& zmq_context, const std::string& port)
    : event_recver(zmq_context, ZMQ_PULL),
      worker_info(std::move(worker_info_)),
      instance_manager(worker_info, port, zmq_context)
    {
        auto bind = "tcp://*:"+port;
        event_recver.bind(bind);
        std::cout << "[MainLoop]: binding to "+bind << std::endl;
    }

    void serve() {
        while (true) {
            int type = zmq_recv_int32(&event_recver);
            std::cout << "[MainLoop]: " << type << std::endl;
            if (type == constants::TASK_TYPE) {
                std::string str = zmq_recv_string(&event_recver);
                std::cout << "[MainLoop]: " << str << std::endl;
                auto bin = zmq_recv_binstream(&event_recver);
                Instance instance;
                bin >> instance;
                instance.show_instance();
                instance_manager.run_instance(instance);
            }
            else if (type == constants::THREAD_FINISHED) {
                int instance_id = zmq_recv_int32(&event_recver);
                int thread_id = zmq_recv_int32(&event_recver);
                instance_manager.finish_thread(instance_id, thread_id);
                bool is_instance_done = instance_manager.is_instance_done(instance_id);
                std::cout << "[MainLoop]: instance_id: " << std::to_string(instance_id) + " tid: " << std::to_string(thread_id) << std::endl;
                if (is_instance_done) {
                    instance_manager.remove_instance(instance_id);
                    std::cout << "[MainLoop]: instance " << std::to_string(instance_id) + "done " << std::endl;
                    // (TODO) tell master
                }
            }
        }
    }

private:
    zmq::socket_t event_recver;
    InstanceManager instance_manager;
    WorkerInfo worker_info;
};

}  // namespace husky
