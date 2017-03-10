#pragma once

#include <thread>
#include <unordered_map>
#include <vector>

#include "core/utility.hpp"

#include "kvpairs.hpp"

#include "husky/base/exception.hpp"
#include "husky/base/serialization.hpp"
#include "husky/core/mailbox.hpp"

#include "kvstore/servercustomer.hpp"
#include "kvstore/handles/basic_server.hpp"
#include "kvstore/handles/ssp_server.hpp"
#include "kvstore/handles/bsp_server.hpp"

#include "kvstore/servercustomer.hpp"

namespace kvstore {

// forward declaration
template <typename Val>
class KVServer;
// template alias
template <typename Val>
using ReqHandle = std::function<void(int, int, husky::base::BinStream&, ServerCustomer*, KVServer<Val>*)>;

/*
 * KVServerBase: Base class for different kvserver
 */
class KVServerBase {
   public:
    virtual void HandleAndReply(int, int, husky::base::BinStream&, ServerCustomer* customer) = 0;
};
template <typename Val>
class KVServer : public KVServerBase {
   public:
    KVServer() = delete;
    KVServer(int kv_id, int server_id, const std::map<std::string, std::string>& hint) {
        try {
            using namespace husky::constants;
            int type = -1;  // 0 for assign, 1 for add, 2 for bsp, 3 for ssp
            if (hint.find(kType) == hint.end()) {  // if kType is not set, use kUpdateType
                if (hint.find(kUpdateType) == hint.end()) {  // default is assign
                    type = 0;
                } else {
                    if (hint.at(kUpdateType) == kAddUpdate) {
                        type = 1;
                    } else if (hint.at(kUpdateType) == kAssignUpdate) {
                        type = 0;
                    } else {
                        throw;
                    }
                }
            } else if (hint.find(kUpdateType) == hint.end()) {  // otherwise set type according to the kType
                if (hint.at(kType) == kSingle 
                        || hint.at(kType) == kHogwild
                        || hint.at(kType) == kSPMT) {
                    type = 0;
                } else if (hint.at(kType) == kPS) {
                    if (hint.at(kConsistency) == kBSP) {
                        type = 2;
                    } else if (hint.at(kConsistency) == kSSP) {
                        type = 3;
                    } else if (hint.at(kConsistency) == kASP) {
                        type = 1;
                    }
                } else {
                    throw;
                }
            } else {
                throw;
            }

            if (hint.find(kStorageType) == hint.end()
                || hint.at(kStorageType) == kUnorderedMapStorage) {  // default is unordered_map
                std::unordered_map<Key, Val> store;
                if (type == 0) {
                    server_base_.reset(new DefaultUpdateServer<Val,
                        std::unordered_map<Key, Val>>(kv_id, server_id, std::move(store), false, true));  // unordered_map, assign
                } else if (type == 1) {
                    server_base_.reset(new DefaultUpdateServer<Val,
                        std::unordered_map<Key, Val>>(kv_id, server_id, std::move(store), false, false));  // unordered_map, add
                } else if (type == 2) {
                    int num_workers = stoi(hint.at(kNumWorkers));
                    server_base_.reset(new BSPServer<Val, 
                            std::unordered_map<Key, Val>>(server_id, num_workers, std::move(store), false));  // unordered_map, bsp
                } else if (type == 3) {
                    int num_workers = stoi(hint.at(kNumWorkers));
                    int staleness = stoi(hint.at(kStaleness));
                    server_base_.reset(new SSPServer<Val, 
                            std::unordered_map<Key, Val>>(server_id, num_workers, std::move(store), false, staleness));  // unordered_map, ssp
                } else {
                    throw;
                }
            } else if (hint.at(husky::constants::kStorageType) == husky::constants::kVectorStorage) {  
                assert(RangeManager::Get().GetMaxKey(kv_id) != std::numeric_limits<Key>::max());
                std::vector<Val> store;
                store.resize(RangeManager::Get().GetServerSize(kv_id, server_id));
                if (type == 0) {
                    server_base_.reset(new DefaultUpdateServer<Val,
                        std::vector<Val>>(kv_id, server_id, std::move(store), true, true));  // vector, assign
                } else if (type == 1) {
                    server_base_.reset(new DefaultUpdateServer<Val,
                        std::vector<Val>>(kv_id, server_id, std::move(store), true, false));  // vector, add
                } else if (type == 2) {
                    int num_workers = stoi(hint.at(kNumWorkers));
                    server_base_.reset(new BSPServer<Val, 
                            std::vector<Val>>(server_id, num_workers, std::move(store), true));  // vector, bsp
                } else if (type == 3) {
                    int num_workers = stoi(hint.at(kNumWorkers));
                    int staleness = stoi(hint.at(kStaleness));
                    server_base_.reset(new 
                            SSPServer<Val, std::vector<Val>>(server_id, num_workers, std::move(store), true, staleness));  // vector, ssp
                } else {
                    throw;
                }
            } else {
                throw;
            }
        } catch (...) {
            husky::utility::print_hint(hint);
            throw husky::base::HuskyException("Unknown KVServer hint");
        }
    }
    ~KVServer() = default;

    /*
     * Handle the BinStream and then reply
     */
    virtual void HandleAndReply(int kv_id, int ts, husky::base::BinStream& bin, ServerCustomer* customer) override {
        server_base_->Process(kv_id, ts, bin, customer);
    }

   private:
    std::unique_ptr<ServerBase> server_base_;
};

/*
 * KVManager manages many KVServer, so different types of data can be stored
 */
class KVManager {
   public:
    KVManager(int server_id, husky::LocalMailbox& mailbox, int channel_id)
        : server_id_(server_id), 
          customer_(new ServerCustomer(
              mailbox, [this](int kv_id, int ts, husky::base::BinStream& bin) { Process(kv_id, ts, bin); },
              channel_id)) {
        customer_->Start();
    }
    ~KVManager() {
        // stop the customer
        customer_->Stop();
        // kv_store_ will be automatically deleted
    }

    /*
     * create different kv_store
     * make sure all the kvstore is set up before the actual workload
     */
    template <typename Val>
    void CreateKVManager(int kv_id, const std::map<std::string, std::string>& hint) {
        kv_store_.insert(std::make_pair(kv_id, std::unique_ptr<KVServer<Val>>(new KVServer<Val>(kv_id, server_id_, hint))));
    }

   private:
    /*
     * Internal receive handle to dispatch the request
     *
     */
    void Process(int kv_id, int ts, husky::base::BinStream& bin) {
        assert(kv_store_.find(kv_id) != kv_store_.end());
        kv_store_[kv_id]->HandleAndReply(kv_id, ts, bin, customer_.get());
    }

    // customer for communication
    std::unique_ptr<ServerCustomer> customer_;
    std::unordered_map<int, std::unique_ptr<KVServerBase>> kv_store_;

    int server_id_;
};

}  // namespace kvstore
