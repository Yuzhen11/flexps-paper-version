#include "cluster_manager/scheduler_trigger.hpp"

#include <chrono>
#include <thread>
#include <iostream> 

#include "husky/core/zmq_helpers.hpp"
#include "husky/base/log.hpp"
#include "core/constants.hpp" 

namespace husky {

SchedulerTrigger::SchedulerTrigger(zmq::context_t* context, std::string cluster_manager_addr) 
    : thread_(nullptr), send_socket_(*context, ZMQ_PUSH) {
    send_socket_.connect(cluster_manager_addr);
}

SchedulerTrigger::~SchedulerTrigger() {
    delete thread_;
    thread_ = nullptr;
}

bool SchedulerTrigger::is_current_ts(int ts) {
    if (expected_timestamp_ == ts) {
        inc_timestamp();
        return true;
    } 
    return false;
}

bool SchedulerTrigger::has_enough_new_threads() {
    counter_ += 1;
    if (counter_ == 1) {
        init_timer();
        return false;
    } else if (counter_ == count_threshold_) {
        inc_timestamp();
        return true;
    }
    return false;
}

void SchedulerTrigger::send_timeout_event() {
    // send the time out schedule event after time_out_period_ seconds
    std::this_thread::sleep_for(std::chrono::seconds(time_out_period_));
    zmq_sendmore_int32(&send_socket_, constants::kClusterManagerTimeOutSchedule);
    zmq_send_int32(&send_socket_, time_out_timestamp_);
    time_out_timestamp_ += 1;
}

void SchedulerTrigger::init_timer() {
    thread_ = new std::thread(&SchedulerTrigger::send_timeout_event, this);
    // let the thread run independently
    thread_->detach();
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
    time_out_period_ = time_out_period;
}

void SchedulerTrigger::set_count_threshold(int threshold) {
    count_threshold_ = threshold;
}

} // namespace husky
