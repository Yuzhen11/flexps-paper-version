#include "core/color.hpp"
#include "husky/core/channel/push_channel.hpp"
#include "husky/core/objlist.hpp"
#include "worker/engine.hpp"

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

    auto task = TaskFactory::Get().CreateTask<HuskyTask>(1, 4);
    engine.AddTask(task, [](const Info& info) {
        int num_pts_per_thread = 1000;
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        int cnt = 0;
        for (int i = 0; i < num_pts_per_thread; i++) {
            double x = distribution(generator);
            double y = distribution(generator);
            if (x * x + y * y <= 1) {
                cnt += 1;
            }
        }
        ObjList<PIObject> pi_list;
        PushChannel<int, PIObject> ch(&pi_list, &pi_list);

        // mailbox
        auto* mailbox = Context::get_mailbox(info.get_local_id());
        // TODO, Channel depends on too many things, bad! At least worker_info can be deleted
        ch.setup(info.get_local_id(), info.get_global_id(), info.get_worker_info(), mailbox);

        ch.push(cnt, 0);
        ch.flush();
        ch.prepare_messages();
        if (pi_list.get_size() > 0) {
            auto& pi_object = pi_list.get(0);
            int sum = 0;
            for (auto i : ch.get(pi_object))
                sum += i;
            int total_pts = num_pts_per_thread * info.get_num_workers();
            husky::LOG_I << BLUE("Estimated PI :" + std::to_string(4.0 * sum / total_pts));
            husky::LOG_I << YELLOW("Estimated PI :" + std::to_string(4.0 * sum / total_pts));
            husky::LOG_I << GREEN("Estimated PI :" + std::to_string(4.0 * sum / total_pts));
            husky::LOG_I << RED("Estimated PI :" + std::to_string(4.0 * sum / total_pts));
            husky::LOG_I << BLACK("Estimated PI :" + std::to_string(4.0 * sum / total_pts));
            husky::LOG_I << CLAY("Estimated PI :" + std::to_string(4.0 * sum / total_pts));
            husky::LOG_I << PURPLE("Estimated PI :" + std::to_string(4.0 * sum / total_pts));
        }
    });
    engine.Submit();

    auto task2 = TaskFactory::Get().CreateTask<Task>(1, 4);
    engine.AddTask(task2, [](const Info& info) { husky::LOG_I << "task2 running"; });

    engine.Submit();
    engine.Exit();
}
