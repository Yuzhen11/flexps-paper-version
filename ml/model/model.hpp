#pragma once

#include <condition_variable>
#include <mutex>
#include <set>
#include <vector>

#include "core/constants.hpp"

namespace ml {
namespace model {

template<typename Val>
class Model {
   public:
    Model(int model_id, int num_params):
        model_id_(model_id), 
        num_params_(num_params) {}
    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) = 0;
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals, int local_id) = 0;

    // TODO: The API is strange
    virtual void Load(int local_id, const std::string& hint) = 0;
    virtual void Dump(int local_id, const std::string& hint) = 0;

   protected:
    int model_id_;
    int num_params_;
};

}  // namespace model
}  // namespace ml
