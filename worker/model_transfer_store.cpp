#include "worker/model_transfer_store.hpp"

namespace husky {

/*
 * Add a model
 */
void ModelTransferStore::Add(int id, std::vector<float>&& param) {
    std::lock_guard<std::mutex> lck(mtx_);
    model_store_.insert({id, std::move(param)});
}
/*
 * Pop the model
 */
std::vector<float> ModelTransferStore::Pop(int id) {
    std::lock_guard<std::mutex> lck(mtx_);
    assert(model_store_.find(id) != model_store_.end());
    std::vector<float> tmp = std::move(model_store_[id]);
    model_store_.erase(id);
    return std::move(tmp);
}

void ModelTransferStore::Clear() {
    model_store_.clear();
}

}  // namespace husky
