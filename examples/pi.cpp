#include "worker/engine.hpp"
#include "husky/core/objlist.hpp"
#include "husky/core/channel/push_channel.hpp"

using namespace husky;

class PIObject {
   public:
    using KeyT=int;
    int key;
    explicit PIObject(KeyT key) { this->key = key; }
    const int& id() const { return key; }
};

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port"});
    if (!rt) return 1;

    Engine engine;

    auto task = TaskFactory::Get().create_task(Task::Type::HuskyTaskType, 1, 4);
    engine.add_task(std::move(task), [](const Info& info) {
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
        // worker_info
        auto* worker_info = &Context::get_worker_info();
        // TODO, Channel depends on too many things, bad! At least worker_info can be deleted
        ch.setup(info.get_local_id(), info.get_global_id(), *worker_info, mailbox);

        ch.push(cnt, 0);
        ch.flush();
        ch.prepare_messages();
        if (pi_list.get_size() > 0) {
            auto& pi_object = pi_list.get(0);
            int sum = 0;
            for (auto i : ch.get(pi_object))
                sum += i;
            int total_pts = num_pts_per_thread * info.get_num_workers();
            base::log_msg("Estimated PI :"+std::to_string(4.0*sum/total_pts));
        }
    });
    engine.submit();
    
    auto task2 = TaskFactory::Get().create_task(Task::Type::BasicTaskType, 1, 4);
    engine.add_task(std::move(task2), [](const Info& info) {
        base::log_msg("task2 running");
    });

    engine.submit();
    engine.exit();
}
