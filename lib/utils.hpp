#pragma once

#include <chrono>

#include "husky/base/assert.hpp"

namespace husky {

class Timer {
   public:
    explicit Timer(bool start_timer) { if (start_timer) start(); }

    void start() {
        ASSERT_MSG(!on_, "Cannot start a running timer.");
        start_time_ = std::chrono::steady_clock::now();
        on_ = true;
    }

    long long elapsed_time() {
        if (on_) {
            return elapsed_time_ + std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time_).count();
        } else return elapsed_time_;
    }

    void pause() {
        if (on_) {
            elapsed_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time_).count();
        } else {
            husky::LOG_I << "Warning: Invalid pause operation. The timer was not started.";
        }
        on_ = false;
    }

    void reset() {
        elapsed_time_ = 0;
        on_ = false;
    }

   private:
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    long long elapsed_time_ = 0;
    bool on_ = false;
};

}  // namespace husky
