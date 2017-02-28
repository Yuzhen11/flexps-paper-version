#pragma once
#include "ml/model/integral_model.hpp"

namespace ml {
namespace model {

/*
 * IntegralModelWithPtr, for Single/Hogwild
 */
class IntegralModelWithPtr : public IntegralModel {
   public:
    IntegralModelWithPtr(int model_id, int num_params):
        IntegralModel(model_id, num_params) {}

    std::vector<float>* GetParamsPtr() {
        return &params_;
    }
};

/*
 * IntegralLockModelWithPtr, for Single/Hogwild
 */
class IntegralLockModelWithPtr : public IntegralLockModel {
   public:
    IntegralLockModelWithPtr(int model_id, int num_params):
        IntegralLockModel(model_id, num_params) {}

    std::vector<float>* GetParamsPtr() {
        return &params_;
    }
};

}  // namespace model
}  // namespace ml

