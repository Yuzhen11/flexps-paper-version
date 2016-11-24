#pragma once

#include <thread>

namespace husky {

/*
 * Unit: A class to manage the UDF thread
 */
class Unit {
public:
    Unit() = default;
    template<typename Exec>
    Unit(Exec exec)
        : th_(exec) {
    }
    // move only
    Unit(Unit&&) = default;
    Unit& operator=(Unit&& rhs) {
        if (th_.joinable())  // join the previous running thread
            th_.join();
        th_ = std::move(rhs.th_);
    }
    ~Unit() {
        if (th_.joinable())
            th_.join();
    }
private:
    std::thread th_;
};

}  // namespace husky
