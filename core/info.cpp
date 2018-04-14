#include "core/info.hpp"

#include "core/task.hpp"

namespace husky {

int Info::get_total_epoch() const { return task_->get_total_epoch(); }

const std::map<std::string, std::string>& Info::get_hint() const { return task_->get_hint(); }

int const Info::get_task_id() const { return task_->get_id(); }

}  // namespace husky
