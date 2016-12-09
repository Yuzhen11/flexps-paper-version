#pragma once

#include <utility>

#include "ml/common/mlworker.hpp"

#include "kvstore/kvstore.hpp"

namespace ml {
namespace single {

class SingleGenericModel : public common::GenericMLWorker {
public:
    SingleGenericModel() = default;

    template<typename... Args>
    SingleGenericModel(int model_id, int local_id, Args&&... args)
        : model_id_(model_id),
          local_id_(local_id),
          model_(std::forward<Args>(args)...) {
    }

    void print_model() const {
        // debug
        for (int i = 0; i < model_.size(); ++ i)
            husky::base::log_msg(std::to_string(model_[i]));
    }

    /*
     * Get parameters from global kvstore
     */
    virtual void Load() override {
        husky::base::log_msg("[Single] loading");
        husky::base::log_msg("[Single] model_id:"+std::to_string(model_id_)+" local_id:"+std::to_string(local_id_));

        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id_);

        std::vector<int> keys(model_.size());
        for (int i = 0; i < keys.size(); ++ i)
            keys[i] = i;
        int ts = kvworker->Pull(model_id_, keys, &model_);
        kvworker->Wait(model_id_, ts);
        print_model();
    }
    /*
     * Put the parameters to global kvstore
     */
    virtual void Dump() override {
        husky::base::log_msg("[Single] dumping");

        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id_);

        std::vector<int> keys(model_.size());
        for (int i = 0; i < keys.size(); ++ i)
            keys[i] = i;
        int ts = kvworker->Push(model_id_, keys, model_);
        kvworker->Wait(model_id_, ts);
    }
    /*
     * Put/Get APIs
     */
    virtual void Put(int key, float val) override {
        assert(key < model_.size());
        model_[key] = val;
    }
    virtual float Get(int key) override {
        assert(key < model_.size());
        return model_[key];
    }
private:
    std::vector<float> model_;
    int model_id_;
    int local_id_;
};

}  // namespace single
}  // namespace ml
