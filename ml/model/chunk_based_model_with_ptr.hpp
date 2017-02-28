#pragma once

#include "ml/model/chunk_based_model.hpp"

namespace ml {
namespace model {

class ChunkBasedModelWithPtr : public ChunkBasedModel {
   public:
    ChunkBasedModelWithPtr(int model_id, int num_params)
        : ChunkBasedModel(model_id, num_params) {}

    std::vector<std::vector<float>>* GetParamsPtr() {
        return &params_;
    }
};

}  // namespace model
}  // namespace ml
