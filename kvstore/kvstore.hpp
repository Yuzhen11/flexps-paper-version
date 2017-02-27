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

    /*
     * \brief Create a new kvstore
     *
     * @param: the function needed by the KVServer, KVServerDefaultAssignHandle by default
     * @return: kvstore id created
     */
    template <typename Val>
    int CreateKVStore(const std::map<std::string, std::string>& hint = {}, husky::constants::Key max_key = std::numeric_limits<husky::constants::Key>::max(),
            int chunk_size = RangeManager::GetDefaultChunkSize()) {
        assert(is_started_);
        // set the default max key and chunk size
        RangeManager::Get().SetMaxKeyAndChunkSize(kv_id, max_key, chunk_size);  
        for (auto* kvserver : kvservers) {
            kvserver->CreateKVManager<Val>(kv_id, hint);
        }
        for (auto* kvworker : kvworkers) {
            kvworker->AddProcessFunc<Val>(kv_id);
        }
        return kv_id++;
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
    std::vector<husky::LocalMailbox*> kvworker_mailboxes;
    std::vector<KVWorker*> kvworkers;
    // mailbox for kvserver
    std::vector<husky::LocalMailbox*> kvserver_mailboxes;
    std::vector<KVManager*> kvservers;

    int num_processes_ = -1;

    bool is_started_ = false;
};

}  // namespace kvstore
