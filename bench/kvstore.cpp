#include <chrono>
#include <gperftools/profiler.h>

#include "kvstore/kvstore.hpp"
#include "worker/engine.hpp"

#include "core/color.hpp"

using namespace husky;

/*
 *
 * An example to test the kvstore performance
 *
 * num_keys=10000000
 * push_or_pull=push/pull
 * vector_storage=on
 *
 * Push/Pull 100M keys using unordered_map storage is around 8s.
 * Push/Pull 100M keys using unordered_map storage is around 1s.
 */
int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port",
            "num_keys", "push_or_pull", "vector_storage"});
    if (!rt)
        return 1;

    // set num_keys
    int num_keys = std::stoi(Context::get_param("num_keys"));
    // set is_push
    bool is_push;
    if (Context::get_param("push_or_pull") == "push") {
        is_push = true;
        husky::LOG_I << RED("set to push");
    }
    else {
        is_push = false;
        husky::LOG_I << RED("set to pull");
    }

    // set vector_storage
    std::map<std::string, std::string> hint;
    if (Context::get_param("vector_storage") == "on") {
        hint = {{husky::constants::kStorageType, husky::constants::kVectorStorage}};
        husky::LOG_I << RED("set to vector");
    } else {
        husky::LOG_I << RED("set to unordered map");
    }

    auto& engine = Engine::Get();
    // Start the kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context());  // num_servers_per_process

    auto task = TaskFactory::Get().CreateTask<Task>(1, 1);
    int kv1 = kvstore::KVStore::Get().CreateKVStore<float>(hint, num_keys, 100);  // hint, max_key, chunk_size
    // kvstore::KVStore::Get().SetMaxKey(kv1, num_keys);
    engine.AddTask(task, [kv1, num_keys, is_push](const Info& info) {
        auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());

        if (is_push) {
            // test Push
            std::vector<husky::constants::Key> keys(num_keys);
            for (size_t i = 0; i < num_keys; ++ i) keys[i] = i;
            std::vector<float> vals(num_keys);
            // ProfilerStart("/home/yzhuang/play/proj/prof/a.prof");
            auto start_time = std::chrono::steady_clock::now();
            int ts = kvworker->Push(kv1, keys, vals);
            kvworker->Wait(kv1, ts);
            auto end_time = std::chrono::steady_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count();
            husky::LOG_I << "total time to pushing " << num_keys << " elements is " << total_time << " ms" << std::endl;
            // ProfilerStop();
        } else {
            // test Pull
            std::vector<husky::constants::Key> keys(num_keys);
            for (int i = 0; i < num_keys; ++ i) {
                keys[i] = i;
            }
            std::vector<float> vals;
            auto start_time = std::chrono::steady_clock::now();
            int ts = kvworker->Pull(kv1, keys, &vals);
            kvworker->Wait(kv1, ts);
            auto end_time = std::chrono::steady_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count();
            husky::LOG_I << "total time to pulling " << num_keys << " elements is " << total_time << " ms" << std::endl;
        }

    });
    engine.Submit();

    engine.Exit();
    // Stop the kvstore, should stop before mailbox is down
    kvstore::KVStore::Get().Stop();
}
