#include "ml/ps/kv_app.hpp"
#include "worker/engine.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port"});
    if (!rt)
        return 1;

    Engine engine;

    auto task = TaskFactory::Get().create_task(Task::Type::PSTaskType, 1, 4);
    static_cast<PSTask*>(task.get())->set_num_ps_servers(2);
    engine.add_task(std::move(task), [](const Info& info) {
        PSTask* ptask = static_cast<PSTask*>(info.get_task());
        if (info.get_cluster_id() == 0) {
            base::log_msg("server num:" + std::to_string(ptask->get_num_ps_servers()));
            base::log_msg("worker num:" + std::to_string(ptask->get_num_ps_workers()));
        }
        if (ptask->is_worker(info.get_cluster_id())) {
            base::log_msg(std::to_string(info.get_cluster_id()) + ": I am a worker");
            ml::ps::KVWorker<float> kvworker(ml::ps::info2psinfo(info), *Context::get_mailbox(info.get_local_id()));
            int num = 10;
            std::vector<int> keys(num);
            std::vector<float> vals(num);

            for (int i = 0; i < num; ++i) {
                keys[i] = i;
                vals[i] = i;
            }

            // push
            int ts = kvworker.Push(keys, vals);
            // wait
            kvworker.Wait(ts);

            // pull
            std::vector<float> rets;
            kvworker.Wait(kvworker.Pull(keys, &rets));
            for (int i = 0; i < num; ++i) {
                base::log_msg("pull result of key:" + std::to_string(keys[i]) + " is: " + std::to_string(rets[i]));
            }
            kvworker.ShutDown();
        } else if (ptask->is_server(info.get_cluster_id())) {
            base::log_msg(std::to_string(info.get_cluster_id()) + ": I am a server");
            ml::ps::KVServer<float> kvserver(ml::ps::info2psinfo(info), *Context::get_mailbox(info.get_local_id()));
            kvserver.ShutDown();
        }
    });
    engine.submit();
    engine.exit();
}
