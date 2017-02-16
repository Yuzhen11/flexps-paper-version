#pragma once

#include <boost/thread.hpp>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <set>
#include <vector>

#include "core/constants.hpp"
#include "ml/model/model.hpp"
#include "kvstore/kvstore.hpp"
#include "ml/model/load.hpp"
#include "ml/model/dump.hpp"

namespace ml {
namespace model {

class IntegralModel : public Model {
   public:
    IntegralModel(int model_id, int num_params):
        Model(model_id, num_params) {}

    virtual void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        for (size_t i = 0; i < keys.size(); ++i) {
            assert(keys[i] < params_.size());
            params_[keys[i]] += vals[i];
        }
    }

    virtual void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals, int local_id) override {
        vals->resize(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            assert(keys[i] < params_.size());
            (*vals)[i] = params_[keys[i]];
        }
    }

    virtual void Load(int local_id) override {  // coordination and sync are handled by MLWorker
        LoadAllIntegral(local_id, model_id_, num_params_, &params_);
    }

    virtual void Dump(int local_id) override {
        DumpAllIntegral(local_id, model_id_, num_params_, params_);
    }

    std::vector<float>* GetParamsPtr() {
        return &params_;
    }
   protected:
    std::vector<float> params_;
};

class IntegralLockModel : public IntegralModel {
   public:
    IntegralLockModel(int model_id, int num_params):
        IntegralModel(model_id, num_params) {}

    void Push(const std::vector<husky::constants::Key>& keys, const std::vector<float>& vals) override {
        boost::lock_guard<boost::shared_mutex> lock(mtx_);
        IntegralModel::Push(keys, vals);
    }

    void Pull(const std::vector<husky::constants::Key>& keys, std::vector<float>* vals, int local_id) override {
        boost::shared_lock<boost::shared_mutex> lock(mtx_);
        IntegralModel::Pull(keys, vals, local_id);
    }

   protected:
    boost::shared_mutex mtx_;
};

}  // namespace model
}  // namespace ml
