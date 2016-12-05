#pragma once

#include <vector>

namespace ml {
namespace common {

/*
 * It serves as an interface for all the parameter-update model in ML
 *
 * TODO: Now we assume the parameters are float
 */
class GenericMLWorker {
public:
    using Callback = std::function<void()>;
    /*
     * Probably we need an initialize function ?
     */
    // virtual void init();
    /*
     * Push/Pull APIs are very suitable for PS, but may not be suitable for 
     * Hogwild! and Single
     */
    virtual int Push(const std::vector<int>& keys, const std::vector<float>& vals, const Callback& cb = nullptr) {
    }
    virtual int Pull(const std::vector<int>& keys, std::vector<float>* vals, const Callback& cb = nullptr) {
    }

    /*
     * Put/Get APIs
     */
    virtual void Put(int key, float val) {
    }
    virtual float Get(int key) {
    }
};

}  // namespace common
}  // namespace ml
