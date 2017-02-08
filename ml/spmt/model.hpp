#pragma once

#include <condition_variable>
#include <mutex>
#include <set>
#include <vector>

#include "core/constants.hpp"

namespace ml {
namespace spmt {

class Model {
   public:
    Model(int local_id, int model_id, int num_params):
        local_id_(local_id),
        model_id_(model_id), 
        num_params_(num_params) {}
    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) = 0;
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals) = 0;
    virtual void Load() = 0;
    virtual void Dump() = 0;

   protected:
    int local_id_;
    int model_id_;
    int num_params_;
};

}
}  // namespace ml
