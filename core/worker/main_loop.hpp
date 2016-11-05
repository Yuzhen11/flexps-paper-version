#pragma once

#include "base/debug.hpp"
#include "base/log.hpp"

#include "zmq.hpp" 
#include "base/serialization.hpp"
#include "core/common/constants.hpp"
#include "core/common/instance.hpp"
#include "core/worker/instance_manager.hpp"
#include "core/common/worker_info.hpp"
#include "core/common/zmq_helpers.hpp"
#include "core/worker/master_connector.hpp"

namespace husky {

class MainLoop {
public:
    MainLoop() = delete;
    MainLoop(WorkerInfo&& worker_info_, MasterConnector&& master_connector_)
        : worker_info(std::move(worker_info_)),
        master_connector(std::move(master_connector_)),
        instance_manager(worker_info, master_connector){
    }

    void serve() {
        while (true) {
            int type = zmq_recv_int32(&master_connector.get_recv_socket());
            base::log_msg("[MainLoop]: " + std::to_string(type) );
            if (type == constants::TASK_TYPE) {
                std::string str = zmq_recv_string(&master_connector.get_recv_socket());
                base::log_msg("[MainLoop]: "+str);
                auto bin = zmq_recv_binstream(&master_connector.get_recv_socket());
                Instance instance;
                bin >> instance;
                instance.show_instance();
                instance_manager.run_instance(instance);
            }
            else if (type == constants::THREAD_FINISHED) {
                int instance_id = zmq_recv_int32(&master_connector.get_recv_socket());
                int thread_id = zmq_recv_int32(&master_connector.get_recv_socket());
                instance_manager.finish_thread(instance_id, thread_id);
                bool is_instance_done = instance_manager.is_instance_done(instance_id);
                base::log_msg("[MainLoop]: instance_id: " + std::to_string(instance_id) + " tid: " + std::to_string(thread_id) + " finished");
                if (is_instance_done) {
                    instance_manager.remove_instance(instance_id);
                    base::log_msg("[MainLoop]: instance " + std::to_string(instance_id) + " done ");
                    // (TODO) tell master
                }
            }
        }
    }

private:
    MasterConnector master_connector;
    WorkerInfo worker_info;
    InstanceManager instance_manager;
};

}  // namespace husky
