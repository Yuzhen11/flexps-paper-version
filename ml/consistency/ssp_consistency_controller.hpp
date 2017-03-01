#include "consistency_controller.hpp"

#include <mutex>
#include <condition_variable>
#include <vector>

namespace ml {
namespace consistency {

class SSPConsistencyController : public AbstractConsistencyController {
   public:
    /*
     * In SSPConsistencyController, only AfterPush is needed since Push will never be blocked.
     *
     * AfterPush will update the worker_progress_, clock_count_ and min_clock_,
     * may potentially wake up other threads
     */
    virtual void BeforePush(int tid) override {}
    virtual void AfterPush(int tid) override {
        // Acquire lock
        std::unique_lock<std::mutex> lck(mtx_);

        if (tid > worker_progress_.size())
            worker_progress_.resize(tid + 1);
        int progress = worker_progress_[tid];
        if (progress >= clock_count_.size())
            clock_count_.resize(progress + 1);
        clock_count_[progress] += 1;
        if (progress == min_clock_ && clock_count_[min_clock_] == num_local_workers_) {
            min_clock_ += 1;
            // release all pull blocked at min_clock_
            cv_.notify_all();
        }
        worker_progress_[tid] += 1;
    }
    /*
     * In SSPConsistencyController, only BeforePull is needed since Pull won't modify the SSPConsistencyController state
     */
    virtual void BeforePull(int tid) override {
        // Acquire lock
        std::unique_lock<std::mutex> lck(mtx_);

        if (tid >= worker_progress_.size())
            worker_progress_.resize(tid + 1);
        int expected_min_lock = worker_progress_[tid] - staleness_;
        while (expected_min_lock > min_clock_) {
            cv_.wait(lck);
        }
    }
    virtual void AfterPull(int tid) override {}
    virtual void Init(int num_local_workers) override {
        worker_progress_.resize(num_local_workers, 0);
        num_local_workers_ = num_local_workers;
    }
    virtual ConsistencyProtocol GetProtocol() override {
        return ConsistencyProtocol::SSP;
    }
   private:
    int num_local_workers_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<int> worker_progress_;
    std::vector<int> clock_count_;
    int staleness_ = 1;
    int min_clock_ = 0;
};

}  // namespace consistency
}  // namespace ml
