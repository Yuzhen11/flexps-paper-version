#include "core/worker/driver.hpp"
// #include "ml/ps/kv_app.hpp"

using namespace husky;

int main(int argc, char** argv) {
    Context::init_global();
    bool rt = Context::get_config()->init_with_args(argc, argv, {});
    if (!rt) return 1;

    Engine engine;

    PSTask task(0, 1, 4, 1);
    engine.add_task(task, [](Info info){
        PSTask task = get_pstask(info.task);
        base::log_msg(std::to_string(info.cluster_id) + ": server num:" + std::to_string(task.get_num_ps_servers()));
        base::log_msg(std::to_string(info.cluster_id) + ": worker num:" + std::to_string(task.get_num_ps_workers()));
        if (task.is_worker(info.cluster_id)) {
            base::log_msg(std::to_string(info.cluster_id) + ": I am a worker");
        } else if (task.is_server(info.cluster_id)) {
            base::log_msg(std::to_string(info.cluster_id) + ": I am a server");
        }
        // if (PSTask::is_worker(info.local_id)) {
        //     KVWorker<float> kvworker(0);
        //     int num = 10;
        //     std::vector<int> keys(num);
        //     std::vector<float> vals(num);
        //
        //     for (int i = 0; i < num; ++ i) {
        //         keys[i] = i;
        //         vals[i] = i;
        //     }
        //
        //     // push
        //     int ts = kvworker.Push(keys, vals);
        //     // wait
        //     kvworker.Wait(ts);
        //
        //     // pull
        //     std::vector<float> rets;
        //     kvworker.Wait(kvworker.Pull(keys, &rets));
        //     for (int i = 0; i < num; ++ i) {
        //         husky::log_msg("pull result of key:"+std::to_string(keys[i])+" is: "+std::to_string(vals[i]));
        //     }
        //
        // } else if (is_server()) {
        //     KVServer<float> kvserver(0);
        // }
    });
    engine.run();
}
