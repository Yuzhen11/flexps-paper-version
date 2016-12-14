#include "husky/core/engine.hpp"
#include "ml/ps/kv_app.hpp"

using namespace husky;

void func() {
    if (Context::get_global_tid() == 0) {
        ml::ps::PSInfo info;
        info.channel_id = 0;
        info.global_id = 0;
        info.num_global_threads = 2;
        info.num_ps_servers = 1;
        std::unordered_map<int, int> c2g;
        c2g.insert({0, 1});
        c2g.insert({1, 0});
        info.cluster_id_to_global_id = c2g;

        base::log_msg(std::to_string(Context::get_global_tid()) + ": I am a worker");
        ml::ps::KVWorker<float> kvworker(info, *Context::get_mailbox(0));
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
        // for (int i = 0; i < num; ++i) {
        //     base::log_msg("pull result of key:" + std::to_string(keys[i]) + " is: " + std::to_string(rets[i]));
        // }
        kvworker.ShutDown();
    } else {
        base::log_msg(std::to_string(Context::get_global_tid()) + ": I am a server");

        ml::ps::PSInfo info;
        info.channel_id = 0;
        info.global_id = 1;
        info.num_global_threads = 2;
        info.num_ps_servers = 1;
        std::unordered_map<int, int> c2g;
        c2g.insert({0, 1});
        c2g.insert({1, 0});
        info.cluster_id_to_global_id = c2g;

        ml::ps::KVServer<float> kvserver(info, *Context::get_mailbox(1));
        kvserver.ShutDown();
    }
}

int main(int argc, char** argv) {
    if (husky::init_with_args(argc, argv)) {
        husky::run_job(func);
        return 0;
    }
    return 1;
}
