#pragma once

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
    void Add(int id, std::vector<float>&& param) {
        std::lock_guard<std::mutex> lck(mtx_);
        model_store_.insert({id, std::move(param)});
    }
    /*
     * Pop the model
     */
    std::vector<float> Pop(int id) {
        std::lock_guard<std::mutex> lck(mtx_);
        assert(model_store_.find(id) != model_store_.end());
        std::vector<float> tmp = std::move(model_store_[id]);
        model_store_.erase(id);
        return std::move(tmp);
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
