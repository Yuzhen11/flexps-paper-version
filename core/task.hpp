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
        ConfigurableWorkersTaskType,
        AutoParallelismTaskType
    };

    // For serialization usage only
    Task() = default;
    Task(int id, Type type = Type::BasicTaskType) : id_(id), type_(type) {}
    Task(int id, int total_epoch, int num_workers, Type type = Type::BasicTaskType)
        : id_(id), total_epoch_(total_epoch), num_workers_(num_workers), type_(type) {}
    virtual ~Task() {}

    virtual BinStream& serialize(BinStream& bin) const {
        bin << id_ << total_epoch_ << current_epoch_ << num_workers_ << type_ << local_ << dmt_;
    }
    virtual BinStream& deserialize(BinStream& bin) {
        bin >> id_ >> total_epoch_ >> current_epoch_ >> num_workers_ >> type_ >> local_>> dmt_;
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
    inline bool get_local() const { return local_; }
    inline bool get_dmt() const { return dmt_; }

    // setter
    inline void set_id(int id) { id_ = id; }
    inline void set_total_epoch(int total_epoch) { total_epoch_ = total_epoch; }
    inline void set_current_epoch(int current_epoch) { current_epoch_ = current_epoch; }
    inline void set_num_workers(int num_workers) { num_workers_ = num_workers; }
    inline void set_type(Type type) { type_ = type; }
    void set_local() { local_= true; }
    void set_dmt() { dmt_ = true; }

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

    Type type_;                                // task type
    bool local_ = false;  // whehter all threads need to be allocated in the same process
    bool dmt_ = false;  // direct model transfer
};

/*
 * AutoParallelismTask will set the parallelism automatically
 */
class AutoParallelismTask : public Task {
   public:
    AutoParallelismTask() = default;
    AutoParallelismTask(int id) : Task(id) { type_ = Type::AutoParallelismTaskType; }

    void set_epoch_iters(const std::vector<int>& iters) {
        assert(iters.size());
        iters_ = iters;
        set_total_epoch(iters.size());
    }
    void set_epoch_iters_and_batchsizes(const std::vector<int>& iters, const std::vector<int>& batchsizes) {
        assert(iters.size());
        assert(iters.size() == batchsizes.size());
        iters_ = iters;
        batchsizes_ = batchsizes;
        set_total_epoch(iters.size());
    }
    const std::vector<int>& get_epoch_iters() const { return iters_; }
    const std::vector<int>& get_batchsizes() const { return batchsizes_; }

    void set_epoch_lambda(const std::function<void(const Info&, int)>& func) { func_ = func; }
    const auto& get_epoch_lambda() { return func_; }

    void set_current_stage_iters(int n_iters) { current_stage_iters_ = n_iters; }
    int get_current_stage_iters() { return current_stage_iters_; }

    virtual BinStream& serialize(BinStream& bin) const {
        Task::serialize(bin);
        bin << iters_ << current_stage_iters_ << batchsizes_;
    }
    virtual BinStream& deserialize(BinStream& bin) {
        Task::deserialize(bin);
        bin >> iters_ >> current_stage_iters_ >> batchsizes_;
    }

    friend BinStream& operator<<(BinStream& bin, const AutoParallelismTask& task) { return task.serialize(bin); }
    friend BinStream& operator>>(BinStream& bin, AutoParallelismTask& task) { return task.deserialize(bin); }

   private:
    std::vector<int> iters_;
    std::vector<int> batchsizes_;
    int current_stage_iters_ = 0;
    std::function<void(const Info&, int)> func_;
};

/*
 * ConfigurableWorkersTask Task
 *
 */
class ConfigurableWorkersTask : public Task {
   public:
    // For serialization usage only
    ConfigurableWorkersTask() = default;
    ConfigurableWorkersTask(int id) : Task(id) { type_ = Type::ConfigurableWorkersTaskType; }
    ConfigurableWorkersTask(int id, int total_epoch, int num_workers)
        : Task(id, total_epoch, num_workers, Type::ConfigurableWorkersTaskType) {}

    void set_worker_num(const std::vector<int>& worker_num) { worker_num_ = worker_num; }
    void set_worker_num_type(const std::vector<std::string>& worker_num_type) { worker_num_type_ = worker_num_type; }

    std::vector<int> get_worker_num() const { return worker_num_; }
    std::vector<std::string> get_worker_num_type() const { return worker_num_type_; }

    virtual BinStream& serialize(BinStream& bin) const {
        Task::serialize(bin);
        return bin << worker_num_ << worker_num_type_;
    }
    virtual BinStream& deserialize(BinStream& bin) {
        Task::deserialize(bin);
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
