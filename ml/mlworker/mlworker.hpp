#pragma once

#include <functional>
#include <vector>

#include "core/constants.hpp"
#include "core/table_info.hpp"
#include "husky/base/exception.hpp"

namespace ml {
namespace mlworker {

/*
 * It serves as an interface for all the parameter-update model in ML
 */
template<typename Val>
class GenericMLWorker {
   public:
    using Callback = std::function<void()>;

    virtual ~GenericMLWorker() {}
    /*
     * Push/Pull APIs are very suitable for PS, but may not be suitable for
     * Hogwild! and Single
     */
    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<Val>& vals) {
        throw husky::base::HuskyException("Push Not implemented");
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<Val>* vals) {
        throw husky::base::HuskyException("Pull Not implemented");
    }

    virtual void PushChunks(const std::vector<husky::constants::Key>& keys, const std::vector<std::vector<Val>*>& vals) {
        throw husky::base::HuskyException("Push Not implemented");
    }
    virtual void PullChunks(const std::vector<husky::constants::Key>& keys, std::vector<std::vector<Val>*>& vals) {
        throw husky::base::HuskyException("Pull Not implemented");
    }

    /*
     * Version 2 APIs, under experiment
     *
     * These set of APIs is to avoid making a copy for Single/Hogwild!
     */
    // Caution: keys should be remained valid during update
    virtual void Prepare_v2(const std::vector<husky::constants::Key>& keys) {
        throw husky::base::HuskyException("v2 Not implemented");
    }
    virtual Val Get_v2(size_t idx) { throw husky::base::HuskyException("v2 Not implemented"); }
    virtual void Update_v2(size_t idx, Val val) { throw husky::base::HuskyException("v2 Not implemented"); }
    virtual void Update_v2(const std::vector<Val>& vals) { throw husky::base::HuskyException("v2 Not implemented"); }
    virtual void Clock_v2(){};  // only for PS
};

}  // namespace mlworker
}  // namespace ml
