#pragma once

#include <memory>
#include <sstream>
#include <vector>

#include "husky/base/exception.hpp"
#include "husky/base/log.hpp"
#include "husky/base/serialization.hpp"

#include "core/color.hpp"
#include "core/info.hpp"

namespace husky {

/*
 * The general Task
 */
using base::BinStream;
class Task {
   public:
    enum class Type {
        BasicTaskType,
        MLTaskType,
        HuskyTaskType,
        DummyType,
        ConfigurableWorkersTaskType,
        AutoParallelismTaskType
    };

    // For serialization usage only
    Task() = default;
    Task(int id, Type type = Type::DummyType) : id_(id), type_(type) {}
    Task(int id, int total_epoch, int num_workers, Type type = Type::BasicTaskType)
        : id_(id), total_epoch_(total_epoch), num_workers_(num_workers), type_(type) {}
    virtual ~Task() {}

    virtual BinStream& serialize(BinStream& bin) const {
        bin << id_ << total_epoch_ << current_epoch_ << num_workers_ << type_ << hint_;
    }
    virtual BinStream& deserialize(BinStream& bin) {
        bin >> id_ >> total_epoch_ >> current_epoch_ >> num_workers_ >> type_ >> hint_;
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
    const std::map<std::string, std::string>& get_hint() const { return hint_; }

    // setter
    inline void set_id(int id) { id_ = id; }
    inline void set_total_epoch(int total_epoch) { total_epoch_ = total_epoch; }
    inline void set_current_epoch(int current_epoch) { current_epoch_ = current_epoch; }
    inline void set_num_workers(int num_workers) { num_workers_ = num_workers; }
    inline void set_type(Type type) { type_ = type; }
    void set_hint(const std::map<std::string, std::string>& hint) { hint_ = hint; }

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
    std::map<std::string, std::string> hint_;  // the hint
};

/*
 * AutoParallelismTask will set the parallelism automatically
 */
class AutoParallelismTask : public Task {
   public:
    AutoParallelismTask() = default;
    AutoParallelismTask(int id) : Task(id, Type::AutoParallelismTaskType) {}

    void set_epoch_iters(const std::vector<int>& iters) {
        iters_ = iters;
    }

    void set_epoch_lambda(const std::function<void(const Info&, int)>& func) {
        func_ = func;
    }

    virtual BinStream& serialize(BinStream& bin) const {
        Task::serialize(bin);
        bin << iters_;
    }
    virtual BinStream& deserialize(BinStream& bin) {
        Task::deserialize(bin);
        bin >> iters_;
    }
    friend BinStream& operator<<(BinStream& bin, const AutoParallelismTask& task) { return task.serialize(bin); }
    friend BinStream& operator>>(BinStream& bin, AutoParallelismTask& task) { return task.deserialize(bin); }
   private:
    std::vector<int> iters_;
    std::function<void(const Info&, int)> func_;
};

class MLTask : public Task {
   public:
    MLTask() = default;
    MLTask(int id) : Task(id, Type::MLTaskType) {}
    MLTask(int id, int total_epoch, int num_workers, Task::Type type) : Task(id, total_epoch, num_workers, type) {}

    void set_dimensions(int dim) { dim_ = dim; }
    void set_kvstore(int kv_id) { kv_id_ = kv_id; }

    size_t get_dimensions() const { return dim_; }
    int get_kvstore() const { return kv_id_; }

    virtual BinStream& serialize(BinStream& bin) const {
        Task::serialize(bin);
        bin << kv_id_ << dim_;
    }
    virtual BinStream& deserialize(BinStream& bin) {
        Task::deserialize(bin);
        bin >> kv_id_ >> dim_;
    }
    friend BinStream& operator<<(BinStream& bin, const MLTask& task) { return task.serialize(bin); }
    friend BinStream& operator>>(BinStream& bin, MLTask& task) { return task.deserialize(bin); }

   protected:

    int kv_id_ = -1;  // the corresponding kvstore id
    size_t dim_ = -1;  // the parameter dimensions
};

/*
 * ConfigurableWorkersTask Task
 *
 */
class ConfigurableWorkersTask : public MLTask {
   public:
    // For serialization usage only
    ConfigurableWorkersTask() = default;
    ConfigurableWorkersTask(int id) : MLTask(id) { type_ = Type::ConfigurableWorkersTaskType; }
    ConfigurableWorkersTask(int id, int total_epoch, int num_workers)
        : MLTask(id, total_epoch, num_workers, Type::ConfigurableWorkersTaskType) {}

    void set_worker_num(const std::vector<int>& worker_num) { worker_num_ = worker_num; }
    void set_worker_num_type(const std::vector<std::string>& worker_num_type) { worker_num_type_ = worker_num_type; }

    std::vector<int> get_worker_num() const { return worker_num_; }
    std::vector<std::string> get_worker_num_type() const { return worker_num_type_; }

    virtual BinStream& serialize(BinStream& bin) const {
        MLTask::serialize(bin);
        return bin << worker_num_ << worker_num_type_;
    }
    virtual BinStream& deserialize(BinStream& bin) {
        MLTask::deserialize(bin);
        return bin >> worker_num_ >> worker_num_type_;
    }
    friend BinStream& operator<<(BinStream& bin, const ConfigurableWorkersTask& task) { return task.serialize(bin); }
    friend BinStream& operator>>(BinStream& bin, ConfigurableWorkersTask& task) { return task.deserialize(bin); }

   private:
     /**
      * worker_num = [5]
      */
     std::vector<int> worker_num_;
     /**
      * worker_num_type equal "threads_per_worker", run 5 threads per worker
      * worker_num_type equal "threads_per_cluster", run 5 threads per cluster
      * worker_num_type equal "local_threads", run 5 local threads
      * worker_num_type equal "threads_traverse_cluster", run 5 threads per worker by per worker
      * worker_num_type equal "threads_on_worker:2", run 5 threads on worker 2
      */
     std::vector<std::string> worker_num_type_;
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
    case Task::Type::MLTaskType: {  // ML Task
        MLTask* task = new MLTask();
        bin >> *task;
        ret.reset(task);
        break;
    }
    case Task::Type::ConfigurableWorkersTaskType: {
      ConfigurableWorkersTask* task = new ConfigurableWorkersTask();
      bin >> *task;
      ret.reset(task);
      break;
    }
    case Task::Type::AutoParallelismTaskType: {
        AutoParallelismTask* task = new AutoParallelismTask();
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
