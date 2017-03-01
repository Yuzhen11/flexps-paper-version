#include "consistency_controller.hpp"

namespace ml {
namespace consistency {

class ASPConsistencyController : public AbstractConsistencyController {
   public:
    virtual void BeforePush(int tid) override {}
    virtual void AfterPush(int tid) override {}

    virtual void BeforePull(int tid) override {}
    virtual void AfterPull(int tid) override {}

    virtual void Init(int num_local_workers) override {}
    virtual ConsistencyProtocol GetProtocol() override {
        return ConsistencyProtocol::ASP;
    }
};

}  // namespace consistency
}  // namespace ml
