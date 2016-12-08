#pragma once

#include <vector>
#include <sstream>
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
        HuskyTaskType,
        DummyType
    };

    // For serialization usage only
    Task() = default;
    Task(int id, Type type = Type::DummyType)
        : id_(id), type_(type)
    {}
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

    // setter
    inline void set_id(int id) { id_ = id; }
    inline void set_total_epoch(int total_epoch) { total_epoch_ = total_epoch; }
    inline void set_current_epoch(int current_epoch) { current_epoch_ = current_epoch; }
    inline void set_num_workers(int num_workers) { num_workers_ = num_workers; }
    inline void set_type(Type type) { type_ = type; }

    inline void inc_epoch() { current_epoch_ += 1; }

    void show() const {
        std::stringstream ss;
        ss << "Task:" << id_ << " total_epoch:" << total_epoch_ << " current_epoch:" \
            << current_epoch_ << " num_workers:" << num_workers_ << " type:" << static_cast<int>(type_);
        base::log_msg("[Task]: "+ss.str());
    }
protected:
    int id_;

    int total_epoch_ = 1;  // total epoch numbers
    int current_epoch_ = 0;

    int num_workers_;  // num of workers needed to run the job

    Type type_;  // task type
};

/*
 * Parameter Server Model Task
 */
class PSTask : public Task {
public:
    // For serialization usage only
    PSTask() = default;
    PSTask(int id)
        : Task(id, Type::PSTaskType)
    {}
    PSTask(int id, int total_epoch, int num_workers)
        : Task(id, total_epoch, num_workers, Type::PSTaskType)
    {}
    // TODO When we add new class member here, we need to override the serialize and
    // deserialize functions!!!
    friend BinStream& operator<<(BinStream& bin, const PSTask& task) {
        return task.serialize(bin);
    }
    friend BinStream& operator>>(BinStream& bin, PSTask& task) {
        return task.deserialize(bin);
    }

    // getter
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

    // setter
    inline void set_num_ps_servers(int num_servers) {
        num_servers_ = num_servers;
    }
private:
    int num_servers_ = 1;
};

/*
 * Single-threaded Model Task
 */
class SingleTask : public Task {
public:
    // For serialization usage only
    SingleTask() = default;
    SingleTask(int id)
        : Task(id, Type::SingleTaskType)
    {}
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
    // For serialization usage only
    HogwildTask() = default;
    HogwildTask(int id)
        : Task(id, Type::HogwildTaskType)
    {}
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
    // For serialization usage only
    GenericMLTask() = default;
    GenericMLTask(int id) 
        : Task(id, Type::GenericMLTaskType)
    {}
    GenericMLTask(int id, int total_epoch, int num_workers)
        : Task(id, total_epoch, num_workers, Type::GenericMLTaskType) 
    {}

    void set_dimensions(int dim) {
        dim_ = dim;
    }
    void set_running_type(Type type) {
        running_type_ = type;
    }

    int get_dimensions() {
        return dim_;
    }

    Type get_running_type() {
        return running_type_;
    }
    
    virtual BinStream& serialize(BinStream& bin) const {
        Task::serialize(bin);
        bin << running_type_;
    }
    virtual BinStream& deserialize(BinStream& bin) {
        Task::deserialize(bin);
        bin >> running_type_;
    }
    friend BinStream& operator<<(BinStream& bin, const GenericMLTask& task) {
        return task.serialize(bin);
    }
    friend BinStream& operator>>(BinStream& bin, GenericMLTask& task) {
        return task.deserialize(bin);
    }
private:
    int dim_;
    Type running_type_ = Type::DummyType;
};

/*
 * Husky Task
 */
class HuskyTask : public Task {
public:
    // For serialization usage only
    HuskyTask() = default;
    HuskyTask(int id)
        : Task(id, Type::HuskyTaskType)
    {}
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


std::unique_ptr<Task> deserialize(BinStream& bin) {
    Task::Type type;
    bin >> type;
    std::unique_ptr<Task> ret;
    switch (type) {
        case Task::Type::BasicTaskType: {   // Basic Task
            Task* task = new Task();
            bin >> *task;
            ret.reset(task);
            break;
        }
        case Task::Type::HuskyTaskType: {  // Husky Task
            HuskyTask* task = new HuskyTask();
            bin >> *task;
            ret.reset(task);
            break;
        }
        case Task::Type::PSTaskType: {  // PS Task
            PSTask* task = new PSTask();
            bin >> *task;
            ret.reset(task);
            break;
        }
        case Task::Type::HogwildTaskType: {  // Hogwild Task
            HogwildTask* task = new HogwildTask();
            bin >> *task;
            ret.reset(task);
            break;
        }
        case Task::Type::SingleTaskType: {  // Single Task
            SingleTask* task = new SingleTask();
            bin >> *task;
            ret.reset(task);
            break;
        }
        case Task::Type::GenericMLTaskType: {  // GenericML Task
            GenericMLTask* task = new GenericMLTask();
            bin >> *task;
            ret.reset(task);
            break;
        }
        default:
            throw base::HuskyException("Deserializing task error");
    }
    return ret;
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
        tasks.push_back(deserialize(bin));
    }
    return tasks;
}

}  // namespace
}  // namespace task

}  // namespace husky
