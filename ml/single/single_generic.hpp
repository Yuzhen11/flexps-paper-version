#pragma once

#include <chrono>

#include <utility>

#include "ml/common/mlworker.hpp"
#include "ml/model/load.hpp"
#include "ml/model/dump.hpp"

#include "kvstore/kvstore.hpp"

namespace ml {
namespace single {

class SingleGenericWorker : public common::GenericMLWorker {
   public:
    SingleGenericWorker() = default;

    template <typename... Args>
    SingleGenericWorker(int model_id, int local_id, Args&&... args)
        : model_id_(model_id), local_id_(local_id), model_(std::forward<Args>(args)...) {
        husky::LOG_I << CLAY("[Single] model_id: "+std::to_string(model_id)
                +" local_id: "+std::to_string(local_id)
                +" model_size: "+std::to_string(model_.size()));
    }

    void print_model() const {
        // debug
        for (int i = 0; i < model_.size(); ++i)
            husky::LOG_I << std::to_string(model_[i]);
    }

    /*
     * Get parameters from global kvstore
     */
    virtual void Load() override {
        model::LoadAllIntegral(local_id_, model_id_, model_.size(), &model_);
        // print_model();
    }
    /*
     * Put the parameters to global kvstore
     */
    virtual void Dump() override {
        model::DumpAllIntegral(local_id_, model_id_, model_.size(), model_);
    }
    /*
     * Put/Get, Push/Pull APIs
     */
    virtual void Put(husky::constants::Key key, float val) override {
        assert(key < model_.size());
        model_[key] = val;
    }
    virtual float Get(husky::constants::Key key) override {
        assert(key < model_.size());
        return model_[key];
    }

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        assert(keys.size() == vals.size());
        for (size_t i = 0; i < keys.size(); i++) {
            assert(keys[i] < model_.size());
            model_[keys[i]] += vals[i];
        }
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals) override {
        vals->resize(keys.size());
        for (size_t i = 0; i < keys.size(); i++) {
            assert(keys[i] < model_.size());
            (*vals)[i] = model_[keys[i]];
        }
    }

    // For v2
    virtual void Prepare_v2(const std::vector<husky::constants::Key>& keys) override {
        keys_ = const_cast<std::vector<husky::constants::Key>*>(&keys);
    }
    virtual float Get_v2(size_t idx) override { return model_[(*keys_)[idx]]; }
    virtual void Update_v2(size_t idx, float val) override { model_[(*keys_)[idx]] += val; }
    virtual void Update_v2(const std::vector<float>& vals) override {
        assert(vals.size() == keys_->size());
        for (size_t i = 0; i < keys_->size(); ++i) {
            assert((*keys_)[i] < model_.size());
            model_[(*keys_)[i]] += vals[i];
        }
    }

   private:
    std::vector<float> model_;
    int model_id_;
    int local_id_;

    // For v2
    // Pointer to keys
    std::vector<husky::constants::Key>* keys_;
};

}  // namespace single
}  // namespace ml
