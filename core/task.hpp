#pragma once

#include <vector>
#include <memory>

#include "base/serialization.hpp"
#include "base/exception.hpp"

#include "ml/common/mlworker.hpp"

namespace husky {

/*
 * The general Task
 */
using base::BinStream;
class Task {
public:
    enum class Type {
        BasicTaskType,
        PSTaskType,
        HogwildTaskType,
        SingleTaskType,
        GenericMLTaskType,
        HuskyTaskType
    };

    Task() = default;
    Task(int id, int total_epoch, int num_workers, Type type = Type::BasicTaskType)
        : id_(id), 
        total_epoch_(total_epoch),
        num_workers_(num_workers),
        type_(type)
    {}
    virtual ~Task() {}

    virtual BinStream& serialize(BinStream& bin) const {
        bin << id_ << total_epoch_ << current_epoch_ << num_workers_ << type_;
    }
    virtual BinStream& deserialize(BinStream& bin) {
        bin >> id_ >> total_epoch_ >> current_epoch_ >> num_workers_ >> type_;
    }

    /*
     * TODO What I want to is to make this function act like virtual
     * But I met the priority problem, that a derived class will first match
     * the general template version before the one takes the base class reference
     * So, for now, user need to rewrite the friend functions in derived class
     */
    friend BinStream& operator<<(BinStream& bin, const Task& task) {
        return task.serialize(bin);
    }
    friend BinStream& operator>>(BinStream& bin, Task& task) {
        return task.deserialize(bin);
    }

    // getter
    inline int get_id() const { return id_; }
    inline int get_total_epoch() const { return total_epoch_; }
    inline int get_current_epoch() const { return current_epoch_; }
    inline int get_num_workers() const { return num_workers_; }
    inline Type get_type() const {return type_; }

    inline void inc_epoch() { current_epoch_ += 1; }
protected:
    int id_;

    int total_epoch_;  // total epoch numbers
    int current_epoch_ = 0;

    int num_workers_;  // num of workers needed to run the job

    Type type_;  // task type
};

/*
 * Parameter Server Model Task
 */
class PSTask : public Task {
public:
    PSTask() = default;
    PSTask(int id, int total_epoch, int num_workers, int num_servers = 1)
        : num_servers_(num_servers),
          Task(id, total_epoch, num_workers, Type::PSTaskType)
    {}
    // TODO When we add new class member here, we need to override the serialize and
    // deserialize functions!!!
    friend BinStream& operator<<(BinStream& bin, const PSTask& task) {
        return task.serialize(bin);
    }
    friend BinStream& operator>>(BinStream& bin, PSTask& task) {
        return task.deserialize(bin);
    }

    inline int get_num_ps_servers() const {
        return num_servers_;
    }
    inline int get_num_ps_workers() const {
        return num_workers_ - num_servers_;
    }
    inline bool is_worker(int id) const {
        return !is_server(id);
    }
    inline bool is_server(int id) const {
        return id < num_servers_;
    }
private:
    int num_servers_;
};

/*
 * Single-threaded Model Task
 */
class SingleTask : public Task {
public:
    SingleTask() = default;
    SingleTask(int id, int total_epoch, int num_workers)
        : Task(id, total_epoch, num_workers, Type::SingleTaskType) 
    {}
    friend BinStream& operator<<(BinStream& bin, const SingleTask& task) {
        return task.serialize(bin);
    }
    friend BinStream& operator>>(BinStream& bin, SingleTask& task) {
        return task.deserialize(bin);
    }
};

/*
 * Hogwild! Model Task
 */
class HogwildTask : public Task {
public:
    HogwildTask() = default;
    HogwildTask(int id, int total_epoch, int num_workers)
        : Task(id, total_epoch, num_workers, Type::HogwildTaskType) 
    {}
    friend BinStream& operator<<(BinStream& bin, const HogwildTask& task) {
        return task.serialize(bin);
    }
    friend BinStream& operator>>(BinStream& bin, HogwildTask& task) {
        return task.deserialize(bin);
    }
};

/*
 * GenericML Task
 *
 * Can be PS, Hogwild! and Single
 */
class GenericMLTask : public Task {
public:
    GenericMLTask() = default;
    GenericMLTask(int id, int total_epoch, int num_workers)
        : Task(id, total_epoch, num_workers, Type::GenericMLTaskType) 
    {}
    GenericMLTask(const GenericMLTask& rhs)
        : Task(rhs),
          worker(new ml::common::GenericMLWorker(*rhs.worker)) {
    }

    // TODO May need to override the serialize function later
    
    friend BinStream& operator<<(BinStream& bin, const GenericMLTask& task) {
        return task.serialize(bin);
    }
    friend BinStream& operator>>(BinStream& bin, GenericMLTask& task) {
        return task.deserialize(bin);
    }

    std::unique_ptr<ml::common::GenericMLWorker>& get_worker() {
        return worker;
    }
private:
    std::unique_ptr<ml::common::GenericMLWorker> worker;
};

/*
 * Husky Task
 */
class HuskyTask : public Task {
public:
    HuskyTask() = default;
    HuskyTask(int id, int total_epoch, int num_workers)
        : Task(id, total_epoch, num_workers, Type::HuskyTaskType) 
    {}
    friend BinStream& operator<<(BinStream& bin, const HuskyTask& task) {
        return task.serialize(bin);
    }
    friend BinStream& operator>>(BinStream& bin, HuskyTask& task) {
        return task.deserialize(bin);
    }
};

namespace task {
namespace {
// Conversion functions to cast down along the task hierarchy
Task& get_task(const std::shared_ptr<Task>& task) {
    return *task.get();
}
PSTask& get_pstask(const std::shared_ptr<Task>& task) {
    return *dynamic_cast<PSTask*>(task.get());
}
HuskyTask& get_huskytask(const std::shared_ptr<Task>& task) {
    return *dynamic_cast<HuskyTask*>(task.get());
}
HogwildTask& get_hogwildtask(const std::shared_ptr<Task>& task) {
    return *dynamic_cast<HogwildTask*>(task.get());
}
GenericMLTask& get_genericmltask(const std::shared_ptr<Task>& task) {
    return *dynamic_cast<GenericMLTask*>(task.get());
}

/*
 * Serialize task bin to std::vector<std::shared_ptr<tasks>>
 *
 * Invoke by Master::recv_tasks_from_worker()
 */
std::vector<std::shared_ptr<Task>> extract_tasks(BinStream& bin) {
    std::vector<std::shared_ptr<Task>> tasks;
    size_t num_tasks;
    bin >> num_tasks;
    for (int i = 0; i < num_tasks; ++ i) {
        Task::Type type;
        bin >> type;
        switch (type) {
            case Task::Type::BasicTaskType: {   // Basic Task
                Task* task = new Task();
                bin >> *task;
                tasks.emplace_back(task);
                break;
            }
            case Task::Type::HuskyTaskType: {  // Husky Task
                HuskyTask* task = new HuskyTask();
                bin >> *task;
                tasks.emplace_back(task);
                break;
            }
            case Task::Type::PSTaskType: {  // PS Task
                PSTask* task = new PSTask();
                bin >> *task;
                tasks.emplace_back(task);
                break;
            }
            case Task::Type::HogwildTaskType: {  // Hogwild Task
                HogwildTask* task = new HogwildTask();
                bin >> *task;
                tasks.emplace_back(task);
                break;
            }
            case Task::Type::GenericMLTaskType: {  // GenericML Task
                GenericMLTask* task = new GenericMLTask();
                bin >> *task;
                tasks.emplace_back(task);
                break;
            }
            default:
                throw base::HuskyException("Deserializing task error");
        }
    }
    return tasks;
}

}  // namespace
}  // namespace task

}  // namespace husky
