#include <vector>

#include "worker/engine.hpp"
#include "ml/common/mlworker.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port"});
    if (!rt) return 1;

    Engine engine;
    // Didn't specify the epoch num and thread num, leave master to decide them
    int kv0 = engine.create_kvstore<float>();
    auto task1 = TaskFactory::Get().create_task(Task::Type::GenericMLTaskType);  
    static_cast<GenericMLTask*>(task1.get())->set_dimensions(10);
    static_cast<GenericMLTask*>(task1.get())->set_running_type(Task::Type::HogwildTaskType);
    task1->set_total_epoch(2);
    engine.add_task(std::move(task1), [](const Info& info){
        auto& worker = info.mlworker;
        int k = 3;
        worker->Put(k, 0.456);
        float v = worker->Get(k);
        base::log_msg("k: "+std::to_string(k) + " v: "+std::to_string(v));
    });

    engine.submit();
    engine.exit();
}

