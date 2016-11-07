#pragma once

namespace husky {

class Task {
public:
    Task() = default;
    Task(int id_, int total_epoch_, int num_workers_)
        : id(id_),
        total_epoch(total_epoch_),
        num_workers(num_workers_)
    {}

    int get_task_id() const {
        return id;
    }

    int id;

    int total_epoch;  // total epoch numbers
    int current_epoch = 0;

    int num_workers;  // num of workers needed to run the job
};

}  // namespace husky
