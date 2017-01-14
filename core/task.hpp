#pragma once

#include <memory>
#include <sstream>
#include <vector>

#include "husky/base/exception.hpp"
#include "husky/base/log.hpp"
#include "husky/base/serialization.hpp"

#include "ml/common/mlworker.hpp"

#include "core/color.hpp"

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
        PSASPTaskType,
        PSSSPTaskType,
        PSBSPTaskType,
        HogwildTaskType,
        SingleTaskType,
        GenericMLTaskType,
        HuskyTaskType,
        DummyType,
        TwoPhasesTaskType
    };

    // For serialization usage only
    Task() = default;
    Task(int id, Type type = Type::DummyType) : id_(id), type_(type) {}
    Task(int id, int total_epoch, int num_workers, Type type = Type::BasicTaskType)
        : id_(id), total_epoch_(total_epoch), num_workers_(num_workers), type_(type) {}
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
    friend BinStream& operator<<(BinStream& bin, const Task& task) { return task.serialize(bin); }
    friend BinStream& operator>>(BinStream& bin, Task& task) { return task.deserialize(bin); }

    // getter
    inline int get_id() const { return id_; }
    inline int get_total_epoch() const { return total_epoch_; }
    inline int get_current_epoch() const { return current_epoch_; }
    inline int get_num_workers() const { return num_workers_; }
    inline Type get_type() const { return type_; }

    // setter
    inline void set_id(int id) { id_ = id; }
    inline void set_total_epoch(int total_epoch) { total_epoch_ = total_epoch; }
    inline void set_current_epoch(int current_epoch) { current_epoch_ = current_epoch; }
    inline void set_num_workers(int num_workers) { num_workers_ = num_workers; }
    inline void set_type(Type type) { type_ = type; }

    inline void inc_epoch() { current_epoch_ += 1; }

    void show() const {
        std::stringstream ss;
        ss << "Task:" << id_ << " total_epoch:" << total_epoch_ << " current_epoch:" << current_epoch_
           << " num_workers:" << num_workers_ << " type:" << static_cast<int>(type_);
        husky::LOG_I << GREEN("[Task]: " + ss.str());
    }

   protected:
    int id_;

    int total_epoch_ = 1;  // total epoch numbers
    int current_epoch_ = 0;

    int num_workers_ = 0;  // num of workers needed to run the job

    Type type_;  // task type
};

class MLTask : public Task {
   public:
    void set_dimensions(int dim) { dim_ = dim; }
    void set_kvstore(int kv_id) { kv_id_ = kv_id; }

    int get_dimensions() { return dim_; }
    int get_kvstore() { return kv_id_; }

   protected:
    // For serialization usage only
    MLTask() = default;
    MLTask(int id, Task::Type type) : Task(id, type) {}
    MLTask(int id, int total_epoch, int num_workers, Task::Type type) : Task(id, total_epoch, num_workers, type) {}

    int kv_id_ = -1;
    int dim_ = -1;
};

/*
 * Deprecated
 *
 * Parameter Server Model Task
 */
class PSTask : public MLTask {
   public:
    // For serialization usage only
    PSTask() = default;
    PSTask(int id) : MLTask(id, Type::PSTaskType) {}
    PSTask(int id, int total_epoch, int num_workers) : MLTask(id, total_epoch, num_workers, Type::PSTaskType) {}
    // TODO When we add new class member here, we need to override the serialize and
    // deserialize functions!!!
    friend BinStream& operator<<(BinStream& bin, const PSTask& task) { return task.serialize(bin); }
    friend BinStream& operator>>(BinStream& bin, PSTask& task) { return task.deserialize(bin); }

    // getter
    inline int get_num_ps_servers() const { return num_servers_; }
    inline int get_num_ps_workers() const { return num_workers_ - num_servers_; }
    inline bool is_worker(int id) const { return !is_server(id); }
    inline bool is_server(int id) const { return id < num_servers_; }

    // setter
    inline void set_num_ps_servers(int num_servers) { num_servers_ = num_servers; }

   private:
    int num_servers_ = 1;
};

/*
 * PSGenericTask,
 * type can be PSBSPTaskType, PSSSPTaskType, PSASPTaskType
 */
class PSGenericTask : public MLTask {
   public:
    // For serialization usage only
    PSGenericTask() = default;
    PSGenericTask(int id, Type type) : MLTask(id, type) {}
    friend BinStream& operator<<(BinStream& bin, const PSGenericTask& task) { return task.serialize(bin); }
    friend BinStream& operator>>(BinStream& bin, PSGenericTask& task) { return task.deserialize(bin); }
};

/*
 * Single-threaded Model Task
 */
class SingleTask : public MLTask {
   public:
    // For serialization usage only
    SingleTask() = default;
    SingleTask(int id) : MLTask(id, Type::SingleTaskType) {}
    SingleTask(int id, int total_epoch, int num_workers) : MLTask(id, total_epoch, num_workers, Type::SingleTaskType) {}
    friend BinStream& operator<<(BinStream& bin, const SingleTask& task) { return task.serialize(bin); }
    friend BinStream& operator>>(BinStream& bin, SingleTask& task) { return task.deserialize(bin); }
};

/*
 * Hogwild! Model Task
 */
class HogwildTask : public MLTask {
   public:
    // For serialization usage only
    HogwildTask() = default;
    HogwildTask(int id) : MLTask(id, Type::HogwildTaskType) {}
    HogwildTask(int id, int total_epoch, int num_workers)
        : MLTask(id, total_epoch, num_workers, Type::HogwildTaskType) {}
    friend BinStream& operator<<(BinStream& bin, const HogwildTask& task) { return task.serialize(bin); }
    friend BinStream& operator>>(BinStream& bin, HogwildTask& task) { return task.deserialize(bin); }
};

/*
 * GenericML Task
 *
 * Can be PS, Hogwild! and Single
 */
class GenericMLTask : public MLTask {
   public:
    // For serialization usage only
    GenericMLTask() = default;
    GenericMLTask(int id) : MLTask(id, Type::GenericMLTaskType) {}
    GenericMLTask(int id, int total_epoch, int num_workers)
        : MLTask(id, total_epoch, num_workers, Type::GenericMLTaskType) {}

    void set_running_type(Type type) { running_type_ = type; }

    Type get_running_type() const { return running_type_; }

    virtual BinStream& serialize(BinStream& bin) const {
        Task::serialize(bin);
        bin << running_type_;
    }
    virtual BinStream& deserialize(BinStream& bin) {
        Task::deserialize(bin);
        bin >> running_type_;
    }
    friend BinStream& operator<<(BinStream& bin, const GenericMLTask& task) { return task.serialize(bin); }
    friend BinStream& operator>>(BinStream& bin, GenericMLTask& task) { return task.deserialize(bin); }

   private:
    Type running_type_ = Type::DummyType;
};

/*
 * TwoPhasesTask
 */
class TwoPhasesTask : public Task {
   public:
    // For serialization usage only
    TwoPhasesTask() = default;
    TwoPhasesTask(int id) : Task(id, Type::TwoPhasesTaskType) {}
    TwoPhasesTask(int id, int total_epoch, int num_workers) : Task(id, total_epoch, num_workers, Type::TwoPhasesTaskType) {}
    friend BinStream& operator<<(BinStream& bin, const TwoPhasesTask& task) { return task.serialize(bin); }
    friend BinStream& operator>>(BinStream& bin, TwoPhasesTask& task) { return task.deserialize(bin); }
};

/*
 * Husky Task
 */
class HuskyTask : public Task {
   public:
    // For serialization usage only
    HuskyTask() = default;
    HuskyTask(int id) : Task(id, Type::HuskyTaskType) {}
    HuskyTask(int id, int total_epoch, int num_workers) : Task(id, total_epoch, num_workers, Type::HuskyTaskType) {}
    friend BinStream& operator<<(BinStream& bin, const HuskyTask& task) { return task.serialize(bin); }
    friend BinStream& operator>>(BinStream& bin, HuskyTask& task) { return task.deserialize(bin); }
};

namespace task {
namespace {
// Conversion functions to cast down along the task hierarchy

std::unique_ptr<Task> deserialize(BinStream& bin) {
    Task::Type type;
    bin >> type;
    std::unique_ptr<Task> ret;
    switch (type) {
    case Task::Type::BasicTaskType: {  // Basic Task
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
    case Task::Type::PSBSPTaskType:
    case Task::Type::PSSSPTaskType:
    case Task::Type::PSASPTaskType: {  // PS Task
        PSGenericTask* task = new PSGenericTask();
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
    case Task::Type::TwoPhasesTaskType: {
      TwoPhasesTask* task = new TwoPhasesTask();
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
 * Invoke by ClusterManager::recv_tasks_from_worker()
 */
std::vector<std::shared_ptr<Task>> extract_tasks(BinStream& bin) {
    std::vector<std::shared_ptr<Task>> tasks;
    size_t num_tasks;
    bin >> num_tasks;
    for (int i = 0; i < num_tasks; ++i) {
        tasks.push_back(deserialize(bin));
    }
    return tasks;
}

}  // namespace
}  // namespace task

}  // namespace husky
