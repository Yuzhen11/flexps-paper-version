#pragma once

#include <vector>
#include <unordered_map>
#include <mutex>

namespace husky {

class ModelTransferStore {
   public:
    static ModelTransferStore& Get() {
        static ModelTransferStore store;
        return store;
    }

    /*
     * Add a model
     */
    void Add(int id, std::vector<float>&& param);

    /*
     * Pop the model
     */
    std::vector<float> Pop(int id);

    /*
     * Clear the ModelTransferStore
     */
    void Clear();

    /*
     * Return the params size
     */
    size_t Size() {
        return model_store_.size();
    }

    ModelTransferStore(const ModelTransferStore&) = delete;
    ModelTransferStore& operator=(const ModelTransferStore&) = delete;
    ModelTransferStore(ModelTransferStore&&) = delete;
    ModelTransferStore& operator=(ModelTransferStore&&) = delete;
   private:
    ModelTransferStore() = default;

    std::unordered_map<int, std::vector<float>> model_store_;
    std::mutex mtx_;
};

}  // namespace husky
