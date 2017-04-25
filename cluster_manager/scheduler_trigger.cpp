#include "cluster_manager/scheduler_trigger.hpp"

#include <chrono>
#include <thread>
#include <iostream> 

#include "husky/core/zmq_helpers.hpp"
#include "husky/base/log.hpp"
#include "core/constants.hpp" 

namespace husky {

SchedulerTrigger::SchedulerTrigger(zmq::context_t* context, std::string cluster_manager_addr) 
    : context_(context), cluster_manager_addr_(cluster_manager_addr) {
}

SchedulerTrigger::~SchedulerTrigger() {
}

bool SchedulerTrigger::is_current_ts(int ts) {
    if (expected_timestamp_ == ts) {
        inc_timestamp();
        return true;
    } 
    return false;
}

bool SchedulerTrigger::has_enough_new_threads() {
    if (counter_ == 0 && enable_timeout_scheduling) {
        // TODO: disable this, cannot spawn threads unlimitedly (cause resource temporarily unavailable)
        // init_timer();
    }
    counter_ += 1;
    if (counter_ == count_threshold_) {
        inc_timestamp();
        return true;
    }
    return false;
}

void SchedulerTrigger::init_timer() {
    thread_ = std::thread([this](){
        std::this_thread::sleep_for(std::chrono::seconds(time_out_period_));
        zmq::socket_t send_socket(*context_, ZMQ_PUSH);
        send_socket.connect(cluster_manager_addr_);
        zmq_sendmore_int32(&send_socket, constants::kClusterManagerTimeOutSchedule);
        std::lock_guard<std::mutex> lock(mu_);
        zmq_send_int32(&send_socket, time_out_timestamp_);
        time_out_timestamp_ += 1;
    });
    // let the thread run independently
    thread_.detach();
}

void SchedulerTrigger::inc_timestamp() {
    expected_timestamp_ += 1;
    counter_ = 0;
}

unsigned int SchedulerTrigger::get_time_out_period() const {
    return time_out_period_;
}

unsigned int SchedulerTrigger::get_count_threshold() const {
    return count_threshold_;
}

void SchedulerTrigger::set_time_out_period(int time_out_period) {
    if (time_out_period == 0) {
        enable_timeout_scheduling = false;
    } else {
        enable_timeout_scheduling = true;
    }
    time_out_period_ = time_out_period;
}

void SchedulerTrigger::set_count_threshold(int threshold) {
    count_threshold_ = threshold;
}

} // namespace husky
