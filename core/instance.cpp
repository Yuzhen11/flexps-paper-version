#include "instance.hpp"

namespace husky {
    
Instance::Instance(const Task& task) { set_task(task); }

void Instance::set_task(const Task& task, const std::string& hint) {
    switch (task.get_type()) {
    case Task::Type::BasicTaskType: {  // Basic Task
        task_.reset(new Task(task));
        break;
    }
    case Task::Type::HuskyTaskType: {  // Husky Task
        task_.reset(new HuskyTask(static_cast<const HuskyTask&>(task)));
        break;
    }
    case Task::Type::MLTaskType: {  // ML Task
        task_.reset(new MLTask(static_cast<const MLTask&>(task)));
        break;
    }
    case Task::Type::ConfigurableWorkersTaskType: { // TwoPhasesTask
        task_.reset(new ConfigurableWorkersTask(static_cast<const ConfigurableWorkersTask&>(task)));
        break;
    }
    default:
        throw base::HuskyException("Constructing instance error");
    }
}

void Instance::show_instance() const {
    task_->show();
    int num_threads = 0;
    for (auto& kv : cluster_)
        num_threads += kv.second.size();
    husky::LOG_I << GREEN("[Instance]: Task id:" + std::to_string(task_->get_id()) + " Epoch:" +
                          std::to_string(task_->get_current_epoch()) + " Proc Num:" +
                          std::to_string(cluster_.size()) + " Thread Num:" + std::to_string(num_threads));
    for (auto& kv : cluster_) {
        std::stringstream ss;
        ss << "Proc id: " << kv.first << ": { ";
        for (auto tid : kv.second) {
            ss << "<" << tid.first << "," << tid.second << "> ";
        }
        ss << "}";
        husky::LOG_I << GREEN("[Instance]: " + ss.str());
    }
}

std::vector<std::pair<int, int>> Instance::get_threads(int proc_id) const {
    auto it = cluster_.find(proc_id);
    if (it == cluster_.end())
        return {};
    else
        return it->second;
}
int Instance::get_num_threads() const {
    int total_threads = 0;
    for (auto& kv : cluster_) {
        total_threads += kv.second.size();
    }
    return total_threads;
}

void Instance::show_instance(int proc_id) const {
    auto iter = cluster_.find(proc_id);
    if (iter == cluster_.end()) {
        husky::LOG_I << GREEN("No instance added in Proc id: " + std::to_string(proc_id));
        return;
    }
    std::stringstream ss;
    task_->show();
    ss << "Task id:" << task_->get_id() << " Proc id:" << iter->first << ": { ";
    for (auto tid : iter->second) {
        ss << "<" << tid.first << "," << tid.second << "> ";
    }
    ss << "}";
    husky::LOG_I << GREEN("[Instance]: " + ss.str() + " Added");
}

BinStream& Instance::serialize(BinStream& bin) const {
    bin << task_->get_type();
    task_->serialize(bin);
    bin << cluster_.size();
    for (auto& kv : cluster_) {
        bin << kv.first << kv.second;
    }
    return bin;
}
BinStream& Instance::deserialize(BinStream& bin) {
    task_ = std::move(task::deserialize(bin));
    size_t size;
    bin >> size;
    cluster_.clear();
    for (size_t i = 0; i < size; ++i) {
        int k;
        std::vector<std::pair<int, int>> v;
        bin >> k >> v;
        cluster_.insert({k, std::move(v)});
    }
    return bin;
}

}  // namespace husky
