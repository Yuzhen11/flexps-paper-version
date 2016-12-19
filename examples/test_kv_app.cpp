#include "ml/ps/kv_app.hpp"
#include "worker/engine.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();

    auto task = TaskFactory::Get().create_task(Task::Type::PSTaskType, 1, 4);
    static_cast<PSTask*>(task.get())->set_num_ps_servers(2);
    engine.AddTask(std::move(task), [](const Info& info) {
        PSTask* ptask = static_cast<PSTask*>(info.get_task());
        if (info.get_cluster_id() == 0) {
            base::log_msg("server num:" + std::to_string(ptask->get_num_ps_servers()));
            base::log_msg("worker num:" + std::to_string(ptask->get_num_ps_workers()));
        }
        if (ptask->is_worker(info.get_cluster_id())) {
            base::log_msg(std::to_string(info.get_cluster_id()) + ": I am a worker");
            ml::ps::KVWorker<float> kv(ml::ps::info2psinfo(info), *Context::get_mailbox(info.get_local_id()));
            int num = 10000;
            std::vector<int> keys(num);
            std::vector<float> vals(num);

            int rank = info.get_global_id();
            srand(rank + 7);
            int kMaxKey = std::numeric_limits<int>::max();
            for (int i = 0; i < num; ++i) {
                keys[i] = kMaxKey / num * i + rank;
                vals[i] = rand() % 1000;
                // vals[i] = 10l;
            }

            // push
            int repeat = 50;
            std::vector<int> ts;
            for (int i = 0; i < repeat; ++i) {
                ts.push_back(kv.Push(keys, vals));
                // to avoid too frequency push, which leads huge memroy usage
                if (i > 10)
                    kv.Wait(ts[ts.size() - 10]);
            }
            for (int t : ts)
                kv.Wait(t);

            // pull
            std::vector<float> rets;
            kv.Wait(kv.Pull(keys, &rets));

            float res = 0;
            for (int i = 0; i < num; ++i) {
                res += fabs(rets[i] - vals[i] * repeat);
            }
            base::log_msg("error: " + std::to_string(res));

            kv.ShutDown();
        } else if (ptask->is_server(info.get_cluster_id())) {
            base::log_msg(std::to_string(info.get_cluster_id()) + ": I am a server");
            ml::ps::KVServer<float> kvserver(ml::ps::info2psinfo(info), *Context::get_mailbox(info.get_local_id()));
            kvserver.ShutDown();
        }
    });
    engine.Submit();
    engine.Exit();
}
