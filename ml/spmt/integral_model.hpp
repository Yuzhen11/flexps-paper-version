#pragma once

#include <boost/thread.hpp>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <set>
#include <vector>

#include "core/constants.hpp"
#include "ml/spmt/model.hpp"
#include "kvstore/kvstore.hpp"

namespace ml {
namespace spmt {

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
        husky::LOG_I << "loading model_id:" + std::to_string(model_id_) + " local_id:" +
                            std::to_string(local_id) + "model_size: " + std::to_string(num_params_);
        auto start_time = std::chrono::steady_clock::now();
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);
        std::vector<husky::constants::Key> keys(num_params_);
        for (size_t i = 0; i < keys.size(); ++i)
            keys[i] = i;
        int ts = kvworker->Pull(model_id_, keys, &params_);
        kvworker->Wait(model_id_, ts);
        auto end_time = std::chrono::steady_clock::now();
        husky::LOG_I << "[Hogwild] Load done and Load time: "
                     << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
                     << " ms";
    }

    virtual void Dump(int local_id) override {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id);

        std::vector<husky::constants::Key> keys(num_params_);
        for (size_t i = 0; i < keys.size(); ++i)
            keys[i] = i;
        int ts = kvworker->Push(model_id_, keys, params_);
        kvworker->Wait(model_id_, ts);
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

}  // namespace spmt
}  // namespace ml
