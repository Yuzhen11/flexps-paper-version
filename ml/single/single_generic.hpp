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
    SingleGenericWorker(int model_id, const husky::Info& info, size_t num_params)
        : info_(info) {
        husky::LOG_I << CLAY("[Single] model_id: "+std::to_string(model_id)
                +" local_id: "+std::to_string(info_.get_local_id())
                +" model_size: "+std::to_string(num_params));
        model_.reset(new model::IntegralModel(model_id, num_params));
        params_ = static_cast<model::IntegralModel*>(model_.get())->GetParamsPtr();
    }

    virtual void Load() override {
        husky::LOG_I << RED("Load epoch: current: "+std::to_string(info_.get_current_epoch()));
        std::string hint;
        if (info_.get_current_epoch() == 0)
            hint = "kvstore";
        else
            hint = "transfer";
        model_->Load(info_.get_local_id(), hint);
    }

    virtual void Dump() override {
        std::string hint;
        husky::LOG_I << RED("Dump epoch: current: "+std::to_string(info_.get_current_epoch())
                +" total: "+std::to_string(info_.get_total_epoch()));
        if (info_.get_current_epoch() < info_.get_total_epoch() - 1)
            hint = "transfer";
        else
            hint = "kvstore";
        model_->Dump(info_.get_local_id(), hint);
    }
    /*
     * Put/Get, Push/Pull APIs
     */
    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        model_->Push(keys, vals);
    }
    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals) override {
        model_->Pull(keys, vals, info_.get_local_id());
    }

    // For v2
    virtual void Prepare_v2(const std::vector<husky::constants::Key>& keys) override {
        keys_ = const_cast<std::vector<husky::constants::Key>*>(&keys);
    }
    virtual float Get_v2(size_t idx) override { return (*params_)[(*keys_)[idx]]; }
    virtual void Update_v2(size_t idx, float val) override { (*params_)[(*keys_)[idx]] += val; }
    virtual void Update_v2(const std::vector<float>& vals) override {
        assert(vals.size() == keys_->size());
        for (size_t i = 0; i < keys_->size(); ++i) {
            assert((*keys_)[i] < params_->size());
            (*params_)[(*keys_)[i]] += vals[i];
        }
    }

   private:
    std::unique_ptr<model::Model> model_;
    // A pointer to the parameter
    std::vector<float>* params_;
    // std::vector<float> model_;
    const husky::Info& info_;

    // For v2
    // Pointer to keys
    std::vector<husky::constants::Key>* keys_;
};

}  // namespace single
}  // namespace ml
