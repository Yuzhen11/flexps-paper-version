#pragma once

#include <chrono>

#include <utility>

#include "ml/common/mlworker.hpp"

#include "kvstore/kvstore.hpp"

namespace ml {
namespace single {

class SingleGenericWorker: public common::GenericMLWorker {
   public:
    SingleGenericWorker() = default;

    template <typename... Args>
    SingleGenericWorker(int model_id, int local_id, Args&&... args)
        : model_id_(model_id), local_id_(local_id), model_(std::forward<Args>(args)...) {}

    void print_model() const {
        // debug
        for (int i = 0; i < model_.size(); ++i)
            husky::LOG_I << std::to_string(model_[i]);
    }

    /*
     * Get parameters from global kvstore
     */
    virtual void Load() override {
        husky::LOG_I << "[Single] loading model_id:" + std::to_string(model_id_) + " local_id:"+
                             std::to_string(local_id_) + "model_size: " + std::to_string(model_.size());
        auto start_time = std::chrono::steady_clock::now();
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id_);
        std::vector<int> keys(model_.size());
        for (int i = 0; i < keys.size(); ++i)
            keys[i] = i;
        int ts = kvworker->Pull(model_id_, keys, &model_);
        kvworker->Wait(model_id_, ts);
        auto end_time = std::chrono::steady_clock::now();
        husky::LOG_I << "[Single] Load done and Load time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count() << " ms";
        //print_model();
    }
    /*
     * Put the parameters to global kvstore
     */
    virtual void Dump() override {
        husky::LOG_I << "[Single] dumping";

        auto* kvworker = kvstore::KVStore::Get().get_kvworker(local_id_);

        std::vector<int> keys(model_.size());
        for (int i = 0; i < keys.size(); ++i)
            keys[i] = i;
        int ts = kvworker->Push(model_id_, keys, model_);
        kvworker->Wait(model_id_, ts);
    }
    /*
     * Put/Get, Push/Pull APIs
     */
    virtual void Put(int key, float val) override {
        assert(key < model_.size());
        model_[key] = val;
    }
    virtual float Get(int key) override {
        assert(key < model_.size());
        return model_[key];
    }

    virtual void Push(const std::vector<int>& keys, const std::vector<float>& vals) override {
        assert(keys.size() == vals.size());
        for (int i = 0; i < keys.size(); i++) {
            assert(keys[i] < model_.size());
            model_[keys[i]] += vals[i];
        }
    }
    virtual void Pull(const std::vector<int>& keys, std::vector<float>* vals) override {
        vals->resize(keys.size());
        for (int i = 0; i < keys.size(); i++) {
            assert(i < model_.size());
            (*vals)[i] = model_[keys[i]];
        }
    }
    

    // For v2
    virtual void Prepare_v2(std::vector<int>& keys) override {
        keys_ = &keys;
    }
    virtual float Get_v2(int idx) override {
        return model_[(*keys_)[idx]];
    }
    virtual void Update_v2(int idx, float val) override {
        model_[(*keys_)[idx]] += val;
    }
    virtual void Update_v2(const std::vector<float>& vals) override {
        assert(vals.size() == keys_->size());
        for (int i = 0; i < keys_->size(); ++ i) {
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
    std::vector<int>* keys_;
};

}  // namespace single
}  // namespace ml
