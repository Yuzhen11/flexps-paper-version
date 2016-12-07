#include <vector>

#include "worker/engine.hpp"
#include "ml/common/mlworker.hpp"

using namespace husky;

int main(int argc, char** argv) {
    Context::init_global();
    bool rt = Context::get_config()->init_with_args(argc, argv, {});
    if (!rt) return 1;

    Engine engine;
    // Didn't specify the epoch num and thread num, leave master to decide them
    auto task1 = TaskFactory::Get().create_task(Task::Type::GenericMLTaskType);  
    engine.add_task(std::move(task1), [](const Info& info){
        auto& worker = info.mlworker;
        int k = 23;
        worker->Put(k, 0.456);
        float v = worker->Get(k);
        base::log_msg("k: "+std::to_string(k) + " v: "+std::to_string(v));
    });

    engine.submit();
    engine.exit();
}

