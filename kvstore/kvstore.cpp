#include "kvstore/kvstore.hpp"

namespace kvstore {

void KVStore::Start(const husky::WorkerInfo& worker_info, husky::MailboxEventLoop* const el, zmq::context_t* zmq_context, int num_servers_per_process) {
    is_started_ = true;
    int num_workers = worker_info.get_num_workers();
    num_processes_ = worker_info.get_num_processes();
    // The following mailboxes [num_workers, 2*num_workers) are for kvworkers
    for (int i = 0; i < num_workers; i++) {
        if (worker_info.get_process_id(i) != worker_info.get_process_id()) {
            el->register_peer_thread(worker_info.get_process_id(i), num_workers + i);  // {proc(i), num_workers+i}
        } else {
            auto mailbox = std::make_unique<husky::LocalMailbox>(zmq_context);
            mailbox->set_thread_id(num_workers + i);
            el->register_mailbox(*mailbox.get());
            kvworker_mailboxes.push_back(std::move(mailbox));
        }
    }

    // The following mailboxes [2*num_workers, 2*num_workers + num_processes_*num_servers_per_process) are for kvservers
    for (int i = 0; i < num_processes_; ++ i) {
        for (int j = 0; j < num_servers_per_process; ++ j) {
            int tid = 2 * num_workers + j * num_processes_ + i;
            if (i != worker_info.get_process_id()) {
                el->register_peer_thread(i, tid);
            } else {
                auto mailbox = std::make_unique<husky::LocalMailbox>(zmq_context);
                mailbox->set_thread_id(tid);
                el->register_mailbox(*mailbox.get());
                kvserver_mailboxes.push_back(std::move(mailbox));
            }
        }
    }

    // Create Servers
    for (int i = 0; i < num_servers_per_process; ++ i) {
        int server_id = i * num_processes_ + worker_info.get_process_id();
        kvservers.push_back(new kvstore::KVManager(*kvserver_mailboxes[i].get(), husky::constants::kv_channel_id));
        server_ids.push_back(server_id);
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
            kvworkers.push_back(new kvstore::KVWorker(info, *kvworker_mailboxes[k].get()));
            k += 1;
        }
    }
    // Set the Rangemanager's NumServers, so RangeManager is ready
    RangeManager::Get().SetNumServers(kvservers.size()*num_processes_);
}
/*
 * \brief kvstore stop function
 */
void KVStore::Stop() {
    is_started_ = false;
    kv_id = 0;
    num_processes_ = -1;
    // 1. delete the kvworkers
    for (auto* p : kvworkers) {
        delete p;
    }
    kvworkers.clear();
    kvworker_mailboxes.clear();
    // 2. delete the kvservers
    for (auto* p : kvservers) {
        delete p;
    }
    kvservers.clear();
    kvserver_mailboxes.clear();
    server_ids.clear();
    // Clear the RangeManager
    RangeManager::Get().Clear();
}

}  // namespace kvstore
