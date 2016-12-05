#pragma once

#include <utility>
#include "core/info.hpp"

#include "ml/common/mlworker.hpp"

namespace ml {
namespace single {

class SingleGenericModel : public common::GenericMLWorker {
public:
    SingleGenericModel() = default;

    template<typename... Args>
    SingleGenericModel(Args&&... args)
        : model(std::forward<Args>(args)...) {
    }

    /*
     * Put/Get APIs
     */
    virtual void Put(int key, float val) override {
        if (key >= model.size())
            model.resize(key+1);
        model[key] = val;
    }
    virtual float Get(int key) override {
        if (key >= model.size())
            model.resize(key+1);
        return model[key];
    }
private:
    std::vector<float> model;
};

}  // namespace single
}  // namespace ml
