#pragma once

#include <mutex>
#include <condition_variable>

namespace ml {
namespace spmt {

enum class ConsistencyProtocol {
    BSP, SSP, ASP
};

class AbstractConsistencyController {
   public:
    virtual void BeforePush(int tid) = 0;
    virtual void AfterPush(int tid) = 0;
    virtual void BeforePull(int tid) = 0;
    virtual void AfterPull(int tid) = 0;
    // one of the workers invoke init
    virtual void Init(int num_local_workers) = 0;
    virtual ConsistencyProtocol GetProtocol() = 0;
};

}  // namespace spmt
}  // namespace ml
