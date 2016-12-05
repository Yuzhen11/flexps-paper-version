#include "worker/engine.hpp"
#include "ml/ps/kv_app.hpp"

using namespace husky;

int main(int argc, char** argv) {
    Context::init_global();
    bool rt = Context::get_config()->init_with_args(argc, argv, {});
    if (!rt) return 1;

    Engine engine;

    PSTask task(0, 1, 4, 2);
    engine.add_task(task, [](Info info){
        PSTask& task = task::get_pstask(info.task);
        if (info.cluster_id == 0) {
            base::log_msg("server num:" + std::to_string(task.get_num_ps_servers()));
            base::log_msg("worker num:" + std::to_string(task.get_num_ps_workers()));
        }
        if (task.is_worker(info.cluster_id)) {
            base::log_msg(std::to_string(info.cluster_id) + ": I am a worker");
            ml::ps::KVWorker<float> kvworker(ml::ps::info2psinfo(info), *Context::get_mailbox(info.local_id));
            int num = 10;
            std::vector<int> keys(num);
            std::vector<float> vals(num);

            for (int i = 0; i < num; ++ i) {
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
            for (int i = 0; i < num; ++ i) {
                base::log_msg("pull result of key:"+std::to_string(keys[i])+" is: "+std::to_string(rets[i]));
            }
            kvworker.ShutDown();
        } else if (task.is_server(info.cluster_id)) {
            base::log_msg(std::to_string(info.cluster_id) + ": I am a server");
            ml::ps::KVServer<float> kvserver(ml::ps::info2psinfo(info), *Context::get_mailbox(info.local_id));
            kvserver.ShutDown();
        }
    });
    engine.submit();
    engine.exit();
}
