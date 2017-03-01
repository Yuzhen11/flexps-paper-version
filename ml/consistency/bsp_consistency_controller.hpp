#include "consistency_controller.hpp"

#include <mutex>
#include <condition_variable>
#include <vector>

namespace ml {
namespace consistency {

class BSPConsistencyController : public AbstractConsistencyController {
   public:
    virtual void BeforePush(int tid) override {
        // Acquire lock
        std::unique_lock<std::mutex> lck(mtx_);
        while (reply_phase_) {
            cv_.wait(lck);
        }
    }
    virtual void AfterPush(int tid) override {
        // Acquire lock
        std::unique_lock<std::mutex> lck(mtx_);
        push_count_ += 1;
        // if all the push are collected, reply for the pull
        if (push_count_ == num_local_workers_) {
            push_count_ = 0;
            reply_phase_ = true;
            // release all the blocked pull
            cv_.notify_all();
        }
    }
    virtual void BeforePull(int tid) override {
        // Acquire lock
        std::unique_lock<std::mutex> lck(mtx_);
        while (!reply_phase_) {
            cv_.wait(lck);
        }
    }
    virtual void AfterPull(int tid) override {
        // Acquire lock
        std::unique_lock<std::mutex> lck(mtx_);
        pull_count_ += 1;
        if (pull_count_ == num_local_workers_) {
            pull_count_ = 0;
            reply_phase_ = false;
            // release all the blocked push
            cv_.notify_all();
        }
    }
    virtual void Init(int num_local_workers) override {
        num_local_workers_ = num_local_workers;
    }
    virtual ConsistencyProtocol GetProtocol() override {
        return ConsistencyProtocol::BSP;
    }
   private:
    int num_local_workers_;
    int push_count_ = 0;
    int pull_count_ = 0;

    bool reply_phase_ = true;

    std::mutex mtx_;
    std::condition_variable cv_;
};

}  // namespace consistency
}  // namespace ml
