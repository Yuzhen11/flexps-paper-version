#pragma once

#include <vector>
#include <thread>

#include "husky/core/zmq_helpers.hpp"

namespace husky {

class SchedulerTrigger {
   public:
    SchedulerTrigger(zmq::context_t* context, std::string cluster_manager_addr);
    ~SchedulerTrigger();

    /*
     * Check whether the given timestamp is the same with current timestamp
     * 1. If ts equals to current timestamp, call inc_timestamp, return true
     * 2. Otherwise return false, cluster manager will ignore this event
     */ 
    bool is_current_ts(int ts);

    /*
     *
     * Increase the counter first and Check whether have enough threads
     * 1. If no previous threads, start a timer, return false
     * 2. If threads number meets threshold, inrease timestamp and restart counter, return true
     * 3. Otherwise, return false
     */ 
    bool has_enough_new_threads();


    unsigned int get_time_out_period() const;
    unsigned int get_count_threshold() const;
    void set_count_threshold(int count_threshold);
    void set_time_out_period(int time);

   private:
    // the function the detached thread runs
    void send_timeout_event();

    // reset counter_ to 0 and increase the timestamp() by 1
    void inc_timestamp();

    // a detached thread is used to generate time out event 
    void init_timer();

   private:
    // count num of newly available threads
    unsigned int counter_ = 0;

    // time of timeout period
    unsigned int time_out_period_ = 5;

    // the threshold to start scheduling on receving enough newly availble threads
    unsigned int count_threshold_= 1;

    // the expected time out timestamp,
    // it increases every successful scheduling
    unsigned int expected_timestamp_ = 0;

    // timestamp for the time out event sent by the detached thread
    // ClusterManager starts scheduling when expected_timestamp_ is equal to time_out_timestamp_
    unsigned int time_out_timestamp_ = 0;

    // used to send timeout scheduling event to cluster_mangager
    zmq::socket_t send_socket_;

    // this thread is detached to run a timer
    std::thread thread_;

    // the default is disable the timeout scheduling
    bool enable_timeout_scheduling = false;
};

} // namespace husky
