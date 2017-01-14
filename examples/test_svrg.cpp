#include "husky/core/channel/push_channel.hpp"
#include "husky/core/objlist.hpp"
#include "worker/engine.hpp"
#include "core/color.hpp"

#include <random>

using namespace husky;

class PIObject {
   public:
    using KeyT = int;
    int key;
    explicit PIObject(KeyT key) { this->key = key; }
    const int& id() const { return key; }
};

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();

    auto task = TaskFactory::Get().CreateTask<TwoPhasesTask>(5, 4);
    engine.AddTask(task, [](const Info& info) {
      const auto& current_task = static_cast<TwoPhasesTask*>(info.get_task());
      husky::LOG_I << RED("current_epoch: " + std::to_string(info.get_current_epoch()));
      if (info.get_current_epoch() % 2 == 1) {
        husky::LOG_I << "--Running odd epoch";
      } else {
        husky::LOG_I << "--Running even epoch";
      } 
      
    });
    engine.Submit();

    engine.Exit();
}
