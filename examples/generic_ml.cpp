#include <vector>

#include "worker/engine.hpp"
#include "ml/common/mlworker.hpp"

using namespace husky;

int main(int argc, char** argv) {
    Context::init_global();
    bool rt = Context::get_config()->init_with_args(argc, argv, {});
    if (!rt) return 1;

    Engine engine;

    GenericMLTask task(0, 1, 4);
    engine.add_task(task, [](Info info){
        GenericMLTask& task = task::get_genericmltask(info.task);

        auto& worker = task.get_worker();
        int k = 23;
        worker->Put(k, 0.456);
        float v = worker->Get(k);
        base::log_msg("k: "+std::to_string(k) + " v: "+std::to_string(v));
    });

    engine.submit();
    engine.exit();
}

