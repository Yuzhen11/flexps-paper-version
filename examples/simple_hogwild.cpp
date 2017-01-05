#include <vector>

#include "ml/hogwild/hogwild.hpp"
#include "worker/engine.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();

    auto task = TaskFactory::Get().CreateTask<HogwildTask>(1, 4);
    engine.AddTask(task, [](const Info& info) {
        int dim = 5;
        // create a hogwild model, which means it's shared
        ml::hogwild::HogwildModel<std::vector<float>> model(*Context::get_zmq_context(), info, dim);
        std::vector<float>* p_model = model.get();
        // update p according to your data
        int j = info.get_cluster_id();
        for (int i = 0; i < 10000; ++i) {
            (*p_model)[j] += 0.01;
            j += 1;
            j %= p_model->size();
        }
        model.sync();
        if (info.get_cluster_id() == 0) {
            for (int i = 0; i < p_model->size(); ++i) {
            husky::LOG_I << (*p_model)[i];
            }
        }

    });
    engine.Submit();
    engine.Exit();
}
