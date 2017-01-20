#pragma once
#include <vector>

#include "core/constants.hpp"
#include "husky/core/mailbox.hpp"
#include "husky/core/worker_info.hpp"
#include "kvmanager.hpp"
#include "kvworker.hpp"

#include "handles/basic_server.hpp"
#include "handles/ssp_server.hpp"
#include "handles/bsp_server.hpp"

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

    /*
     * \brief kvstore start function
     *
     * Create new mailboxes and add them to el
     */
    void Start(const husky::WorkerInfo& worker_info, husky::MailboxEventLoop* const el, zmq::context_t* zmq_context) {
        is_started_ = true;
        int num_workers = worker_info.get_num_workers();
        int num_processes = worker_info.get_num_processes();
        // The following mailboxes [num_workers - 2*num_workers) are for kvworkers
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

        // The following mailboxes [2*num_workers - 2*num_workers+num_processes) are for kvservers
        // Each process has one kvmanager by default
        for (int i = 0; i < num_processes; ++i) {
            int tid = 2 * num_workers + i;
            if (i != worker_info.get_process_id()) {
                el->register_peer_thread(i, tid);
            } else {
                auto* mailbox = new husky::LocalMailbox(zmq_context);
                mailbox->set_thread_id(tid);
                el->register_mailbox(*mailbox);
                kvserver_mailbox = mailbox;
            }
        }

        // Create kvmanager
        kvmanager.reset(new kvstore::KVManager(*kvserver_mailbox, husky::constants::kv_channel_id));

        // Create kvworkers
        std::unordered_map<int, int> cluster2global;
        for (int i = 0; i < num_processes; ++i) {
            cluster2global.insert({i, i + 2 * num_workers});
        }
        for (int i = 0; i < num_workers; ++i) {
            cluster2global.insert({i + num_processes, i + num_workers});
        }
        int k = 0;
        for (int i = 0; i < num_workers; ++i) {
            if (worker_info.get_process_id(i) == worker_info.get_process_id()) {
                kvstore::PSInfo info;
                info.channel_id = husky::constants::kv_channel_id;
                info.global_id = num_workers + i;
                info.num_global_threads = num_workers + num_processes;  // workers + servers
                info.num_ps_servers = num_processes;                    // TODO: local_kvstore only need one server
                info.cluster_id_to_global_id = cluster2global;
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
        // 1. delete the kvworkers
        for (auto* p : kvworkers) {
            delete p;
        }
        // destroy the mailboxes
        for (auto* p : kvworker_mailboxes) {
            delete p;
        }
        // 2. delete the kvmanager
        kvmanager.reset();
        // destroy the mailbox
        delete kvserver_mailbox;
    }

    /*
     * \brief Create a new kvstore
     *
     * @param: the function needed by the KVServer, KVServerDefaultAssignHandle by default
     * @return: kvstore id created
     */
    template <typename Val>
    int CreateKVStore(const ReqHandle<Val>& request_handler = KVServerDefaultAssignHandle<Val>()) {
        assert(is_started_);
        kvmanager->CreateKVManager<Val>(kv_id, request_handler);
        for (auto* kvworker : kvworkers) {
            kvworker->AddProcessFunc<Val>(kv_id);
        }
        return kv_id++;
    }

    void SetMaxKey(int kv_id, husky::constants::Key max_key) {
        for (auto* kvworker : kvworkers) {
            kvworker->SetMaxKey(kv_id, max_key);
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
    std::vector<husky::LocalMailbox*> kvworker_mailboxes;
    std::vector<KVWorker*> kvworkers;
    // mailbox for kvserver
    husky::LocalMailbox* kvserver_mailbox;
    std::unique_ptr<KVManager> kvmanager;

    bool is_started_ = false;
};

}  // namespace kvstore
