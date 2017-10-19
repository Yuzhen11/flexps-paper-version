#pragma once

#include <vector>

#include "core/constants.hpp"
#include "husky/core/mailbox.hpp"
#include "husky/core/worker_info.hpp"
#include "kvmanager.hpp"
#include "kvworker.hpp"
#include "range_manager.hpp"

#include "handles/basic_server.hpp"
#include "handles/bsp_server.hpp"
#include "handles/ssp_server.hpp"

namespace kvstore {

/*
 * KVStore: a singleton to manage the kvmanager and kvworker
 */
class KVStore {
   public:
    /*
     * \brief return the singleton object
     */
    static KVStore& Get() {
        static KVStore kvstore;
        return kvstore;
    }
    KVStore(const KVStore&) = delete;
    KVStore operator=(const KVStore&) = delete;
    KVStore(KVStore&&) = delete;
    KVStore operator=(KVStore&&) = delete;

    /*
     * \brief kvstore start function
     *
     * Create new mailboxes and add them to el
     * The last parameter is to set the number of server threads in each process
     */
    void Start(const husky::WorkerInfo& worker_info, husky::MailboxEventLoop* const el, zmq::context_t* zmq_context, int num_servers_per_process = 1);

    /*
     * \brief kvstore stop function
     */
    void Stop();

    template <typename Val>
    std::unique_ptr<ServerBase> ServerFactory(const std::string& hint, int num_workers, int staleness, int server_id) {
        using Key = husky::constants::Key;
        std::unique_ptr<ServerBase> server;
        if (hint == "default_assign_map") {
            std::unordered_map<Key, Val> store;
            server.reset(new DefaultUpdateServer<Val,
                std::unordered_map<Key, Val>>(kv_id, server_id, std::move(store), false, true));  // unordered_map, assign
        } else if (hint == "default_add_map") {
            std::unordered_map<Key, Val> store;
            server.reset(new DefaultUpdateServer<Val,
                std::unordered_map<Key, Val>>(kv_id, server_id, std::move(store), false, true));  // unordered_map, assign
        } else if (hint == "bsp_add_map") {
            std::unordered_map<Key, Val> store;
            server.reset(new BSPServer<Val, 
                std::unordered_map<Key, Val>>(server_id, num_workers, std::move(store), false, false));  // unordered_map, bsp
        } else if (hint == "ssp_add_map") {
            std::unordered_map<Key, Val> store;
            server.reset(new SSPServer<Val, 
                std::unordered_map<Key, Val>>(server_id, num_workers, std::move(store), false, staleness));  // unordered_map, ssp
        } else if (hint == "default_assign_vector") {
            assert(RangeManager::Get().GetMaxKey(kv_id) != std::numeric_limits<Key>::max());
            std::vector<Val> store;
            store.resize(RangeManager::Get().GetServerSize(kv_id, server_id));
            server.reset(new DefaultUpdateServer<Val,
                std::vector<Val>>(kv_id, server_id, std::move(store), true, true));  // vector, assign
        } else if (hint == "default_add_vector") {
            assert(RangeManager::Get().GetMaxKey(kv_id) != std::numeric_limits<Key>::max());
            std::vector<Val> store;
            store.resize(RangeManager::Get().GetServerSize(kv_id, server_id));
            server.reset(new DefaultUpdateServer<Val,
                std::vector<Val>>(kv_id, server_id, std::move(store), true, false));  // vector, add
        } else if (hint == "bsp_add_vector") {
            assert(RangeManager::Get().GetMaxKey(kv_id) != std::numeric_limits<Key>::max());
            std::vector<Val> store;
            store.resize(RangeManager::Get().GetServerSize(kv_id, server_id));
            server.reset(new BSPServer<Val, 
                std::vector<Val>>(server_id, num_workers, std::move(store), true, false));  // vector, bsp
        } else if (hint == "ssp_add_vector") {
            assert(RangeManager::Get().GetMaxKey(kv_id) != std::numeric_limits<Key>::max());
            std::vector<Val> store;
            store.resize(RangeManager::Get().GetServerSize(kv_id, server_id));
            server.reset(new 
                SSPServer<Val, std::vector<Val>>(server_id, num_workers, std::move(store), true, staleness));  // vector, ssp
        } else {
            husky::LOG_I << "Unknown hint: " << hint;
            assert(false);
        }
        return server;
    }

    /*
     * \brief Create a new kvstore
     *
     * @param max_key max key of hte kvstore
     * @param chunk_size the chunk_size
     * @return kvstore id created
     */
    template <typename Val>
    int CreateKVStore(const std::string& hint, int num_workers, int staleness,
            husky::constants::Key max_key = std::numeric_limits<husky::constants::Key>::max(),
            int chunk_size = RangeManager::GetDefaultChunkSize()) {
        assert(is_started_);
        // set the default max key and chunk size
        RangeManager::Get().SetMaxKeyAndChunkSize(kv_id, max_key, chunk_size);  
        for (int i = 0; i < kvservers.size(); ++ i) {
            assert(i < server_ids.size());
            int server_id = server_ids[i];
            std::unique_ptr<ServerBase> server = ServerFactory<Val>(hint, num_workers, staleness, server_id);
            kvservers[i]->CreateKVManager<Val>(kv_id, std::move(server));
        }
        for (auto* kvworker : kvworkers) {
            kvworker->AddProcessFunc<Val>(kv_id);
        }
        return kv_id++;
    }

    /*
     * \brief only create kvstore, but cannot be used before setup, 
     * This is use for user that need to customize the ranges
     * The only difference is that user can customize the ranges by themselves
     */
    int CreateKVStoreWithoutSetup() {
        assert(is_started_);
        return kv_id++;
    }
    template<typename Val>
    void SetupKVStore(int id, const std::string& hint, int num_workers, int staleness) {
        for (int i = 0; i < kvservers.size(); ++ i) {
            assert(i < server_ids.size());
            int server_id = server_ids[i];
            std::unique_ptr<ServerBase> server = ServerFactory<Val>(hint, num_workers, staleness, server_id);
            kvservers[i]->CreateKVManager<Val>(kv_id, std::move(server));
        }
        for (auto* kvworker : kvworkers) {
            kvworker->AddProcessFunc<Val>(id);
        }
    }

    /*
     * \brief function to return kvworker
     */
    KVWorker* get_kvworker(int i) {
        assert(i >= 0 && i < kvworkers.size());
        return kvworkers[i];
    }

   private:
    KVStore() = default;

    // kv_id counter
    int kv_id = 0;
    // mailboxes for kvworker
    std::vector<std::unique_ptr<husky::LocalMailbox>> kvworker_mailboxes;
    std::vector<KVWorker*> kvworkers;
    // mailbox for kvserver
    std::vector<std::unique_ptr<husky::LocalMailbox>> kvserver_mailboxes;
    std::vector<KVManager*> kvservers;
    std::vector<int> server_ids;

    int num_processes_ = -1;

    bool is_started_ = false;
};

}  // namespace kvstore
