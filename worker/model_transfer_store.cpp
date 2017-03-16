#include "worker/model_transfer_store.hpp"

namespace husky {

/*
 * Add a model
 */
void ModelTransferStore::Add(int id, husky::base::BinStream&& bin) {
    std::lock_guard<std::mutex> lck(mtx_);
    model_store_.insert({id, std::move(bin)});
}
/*
 * Pop the model
 */
husky::base::BinStream ModelTransferStore::Pop(int id) {
    std::lock_guard<std::mutex> lck(mtx_);
    assert(model_store_.find(id) != model_store_.end());
    auto ret= std::move(model_store_[id]);
    model_store_.erase(id);
    return ret;
}

void ModelTransferStore::Clear() {
    model_store_.clear();
}

}  // namespace husky
