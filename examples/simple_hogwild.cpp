#include <vector>

#include "worker/driver.hpp"
#include "ml/hogwild/hogwild.hpp"

using namespace husky;

int main(int argc, char** argv) {
    Context::init_global();
    bool rt = Context::get_config()->init_with_args(argc, argv, {});
    if (!rt) return 1;

    Engine engine;

    HogwildTask task(0, 1, 4);
    engine.add_task(task, [](Info info){
        HogwildTask task = get_hogwildtask(info.task);

        int dim = 100;
        // create a hogwild model, which means it's shared
        ml::hogwild::HogwildModel<std::vector<float>> model(Context::get_zmq_context(), info, dim);
        std::vector<float>* p_model = model.get();
        // update p according to your data
        int j = info.cluster_id;
        for (int i = 0; i < 1000; ++ i) {
            (*p_model)[j] += 1;
            j += 1;
            j %= p_model->size();
        }
        model.sync();
        if (info.cluster_id == 0) {
            for (int i = 0; i < p_model->size(); ++ i) {
                husky::base::log_msg(std::to_string((*p_model)[i]));
            }
        }

    });
    engine.run();
}
