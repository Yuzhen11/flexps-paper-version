#pragma once

#include <iostream>

#include "core/instance.hpp"
#include "core/zmq_helpers.hpp"

#include "zmq.hpp"

namespace husky {

class Assigner {
public:
    Assigner() = default;
    void assign_task(const Instance& instance) {
        // 1. extract dest
        // 2. send the instance to dest
    }

private:
};

}  // namespace husky
