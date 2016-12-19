#include <vector>

#include "ml/hogwild/hogwild.hpp"
#include "worker/engine.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();

    auto task = TaskFactory::Get().create_task(Task::Type::HogwildTaskType, 1, 4);
    engine.AddTask(std::move(task), [](const Info& info) {
        int dim = 100;
        // create a hogwild model, which means it's shared
        ml::hogwild::HogwildModel<std::vector<float>> model(*Context::get_zmq_context(), info, dim);
        std::vector<float>* p_model = model.get();
        // update p according to your data
        int j = info.get_cluster_id();
        for (int i = 0; i < 1000; ++i) {
            (*p_model)[j] += 1;
            j += 1;
            j %= p_model->size();
        }
        model.sync();
        if (info.get_cluster_id() == 0) {
            for (int i = 0; i < p_model->size(); ++i) {
                husky::base::log_msg(std::to_string((*p_model)[i]));
            }
        }

    });
    engine.Submit();
    engine.Exit();
}
