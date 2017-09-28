#include "kvstore/kvstore.hpp"
#include "worker/engine.hpp"
#include "kvstore/ps_lite/sarray.h"
#include "core/color.hpp"

using namespace husky;

int main(int argc, char** argv) {
    bool rt = init_with_args(argc, argv, {"worker_port", "cluster_manager_host", "cluster_manager_port"});
    if (!rt)
        return 1;

    auto& engine = Engine::Get();
    // Start the kvstore, should start after mailbox is up
    kvstore::KVStore::Get().Start(Context::get_worker_info(), Context::get_mailbox_event_loop(),
                                  Context::get_zmq_context(), 2);

    const int kMaxKey = 100000000;

    // Test KVStore using KVServerSSPHandle: SSP
    std::map<std::string, std::string> hint = 
    {
        {husky::constants::kType, husky::constants::kPS},
        {husky::constants::kConsistency, husky::constants::kSSP},
        {husky::constants::kStaleness, "1"},
        {husky::constants::kNumWorkers, "2"},
    };
    auto task = TaskFactory::Get().CreateTask<Task>(1, 2); // one epoch, two workers
    int kv = kvstore::KVStore::Get().CreateKVStore<float>(hint, kMaxKey, kMaxKey / 2);
    engine.AddTask(task, [kv](const Info& info) {

        if (Context::get_process_id() == 0){
            auto* kvworker = kvstore::KVStore::Get().get_kvworker(info.get_local_id());

            std::vector<husky::constants::Key> keys1(kMaxKey / 2);
            std::vector<husky::constants::Key> keys2(kMaxKey / 2);

            std::iota(keys1.begin(), keys1.end(), 0);
            std::iota(keys2.begin(), keys2.end(), kMaxKey / 2);

            std::vector<float> vals(kMaxKey / 2, 0.5);
            std::vector<float> rets;

            // push and pull in the same node
            auto start_time = std::chrono::steady_clock::now();
            kvworker->Wait(kv, kvworker->Pull(kv, keys1, &rets));
            auto end_time = std::chrono::steady_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            husky::LOG_I << YELLOW("Total time of Pull to server in the same node: " + std::to_string(total_time) + " ms");
            
            kvworker->Wait(kv, kvworker->Push(kv, keys1, vals));
            
            end_time = std::chrono::steady_clock::now();
            total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            husky::LOG_I << YELLOW("Total time of Pull and Push to server in the same node: " + std::to_string(total_time) + " ms");


            // push and pull in the different node
            start_time = std::chrono::steady_clock::now();
            kvworker->Wait(kv, kvworker->Pull(kv, keys2, &rets));
            end_time = std::chrono::steady_clock::now();
            total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            husky::LOG_I << YELLOW("Total time of Pull to server in the different node: " + std::to_string(total_time) + " ms");
            
            kvworker->Wait(kv, kvworker->Push(kv, keys2, vals));
            
            end_time = std::chrono::steady_clock::now();
            total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            husky::LOG_I << YELLOW("Total time of Pull and Push to server in the different node: " + std::to_string(total_time) + " ms");
        }
    });
    engine.Submit();
    engine.Exit();
    // Stop the kvstore, should stop before mailbox is down
    kvstore::KVStore::Get().Stop();
}
