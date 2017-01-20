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

    auto task = TaskFactory::Get().CreateTask<FixedWorkersTask>(5, 4);
    engine.AddTask(task, [](const Info& info) {
      husky::LOG_I << "Running TwoFixedWorkersTask!";
    });
    engine.Submit();

    engine.Exit();
}
