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
    void Start(const husky::WorkerInfo& worker_info, husky::MailboxEventLoop* const el, zmq::context_t* zmq_context, int num_servers_per_process = 1) {
        is_started_ = true;
        int num_workers = worker_info.get_num_workers();
        num_processes_ = worker_info.get_num_processes();
        // The following mailboxes [num_workers, 2*num_workers) are for kvworkers
        for (int i = 0; i < num_workers; i++) {
            if (worker_info.get_process_id(i) != worker_info.get_process_id()) {
                el->register_peer_thread(worker_info.get_process_id(i), num_workers + i);  // {proc(i), num_workers+i}
            } else {
                auto* mailbox = new husky::LocalMailbox(zmq_context);
                mailbox->set_thread_id(num_workers + i);
                el->register_mailbox(*mailbox);
                kvworker_mailboxes.push_back(mailbox);
            }
        }

        // The following mailboxes [2*num_workers, 2*num_workers + num_processes_*num_servers_per_process) are for kvservers
        for (int i = 0; i < num_processes_; ++ i) {
            for (int j = 0; j < num_servers_per_process; ++ j) {
                int tid = 2 * num_workers + j * num_processes_ + i;
                if (i != worker_info.get_process_id()) {
                    el->register_peer_thread(i, tid);
                } else {
                    auto* mailbox = new husky::LocalMailbox(zmq_context);
                    mailbox->set_thread_id(tid);
                    el->register_mailbox(*mailbox);
                    kvserver_mailboxes.push_back(mailbox);
                }
            }
        }

        // Create Servers
        for (int i = 0; i < num_servers_per_process; ++ i) {
            int server_id = i * num_processes_ + worker_info.get_process_id();
            kvservers.push_back(new kvstore::KVManager(server_id, *kvserver_mailboxes[i], husky::constants::kv_channel_id));
        }

        // Create kvworkers
        std::unordered_map<int, int> server2global;  // Generate the server id to global id map
        for (int i = 0; i < num_processes_; ++ i) {
            for (int j = 0; j < num_servers_per_process; ++ j) {
                server2global.insert({i + j * num_processes_, 2*num_workers + j * num_processes_ + i});
            }
        }
        int k = 0;
        for (int i = 0; i < num_workers; ++i) {
            if (worker_info.get_process_id(i) == worker_info.get_process_id()) {
                kvstore::PSInfo info;
                info.channel_id = husky::constants::kv_channel_id;
                info.global_id = num_workers + i;
                info.num_ps_servers = num_processes_ * num_servers_per_process;
                for (int j = 0; j < num_servers_per_process; ++ j) {
                    info.local_server_ids.insert(worker_info.get_process_id() + j * num_processes_);
                }
                info.server_id_to_global_id= server2global;
                kvworkers.push_back(new kvstore::KVWorker(info, *kvworker_mailboxes[k]));
                k += 1;
            }
        }
    }
    /*
     * \brief kvstore stop function
     */
    void Stop() {
        is_started_ = false;
        kv_id = 0;
        num_processes_ = -1;
        RangeManager::Get().Clear();
        // 1. delete the kvworkers
        for (auto* p : kvworkers) {
            delete p;
        }
        kvworkers.clear();
        // destroy the mailboxes
        for (auto* p : kvworker_mailboxes) {
            delete p;
        }
        kvworker_mailboxes.clear();
        // 2. delete the kvservers
        for (auto* p : kvservers) {
            delete p;
        }
        kvservers.clear();
        // destroy the mailbox
        for (auto* p : kvserver_mailboxes) {
            delete p;
        }
        kvserver_mailboxes.clear();
    }

    /*
     * \brief Create a new kvstore
     *
     * @param: the function needed by the KVServer, KVServerDefaultAssignHandle by default
     * @return: kvstore id created
     */
    template <typename Val>
    int CreateKVStore(const std::string& hint = "") {
        assert(is_started_);
        // set the default max key and chunk size
        RangeManager::Get().SetNumServers(kvservers.size()*num_processes_);
        RangeManager::Get().SetMaxKeyAndChunkSize(kv_id);  
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
