#pragma once

#include <thread>
#include <unordered_map>
#include <vector>

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
            if (hint.find(husky::constants::kStorageType) == hint.end()
                || hint.at(husky::constants::kStorageType) == husky::constants::kUnorderedMapStorage) {  // default is unordered_map
                std::unordered_map<husky::constants::Key, Val> store;
                if (hint.find(husky::constants::kType) == hint.end()
                        || hint.at(husky::constants::kType) == husky::constants::kSingle
                        || hint.at(husky::constants::kType) == husky::constants::kHogwild
                        || hint.at(husky::constants::kType) == husky::constants::kSPMT) {  // default is assign
                    server_base_.reset(new DefaultUpdateServer<Val,
                        std::unordered_map<husky::constants::Key, Val>>(kv_id, server_id, std::move(store), false, true));  // unordered_map, assign
                } else if (hint.at(husky::constants::kType) == husky::constants::kPS
                        && hint.at(husky::constants::kConsistency) == husky::constants::kASP) {     // unordered_map, add
                    server_base_.reset(new DefaultUpdateServer<Val,
                        std::unordered_map<husky::constants::Key, Val>>(kv_id, server_id, std::move(store), false, false));
                } else if (hint.at(husky::constants::kType) == husky::constants::kPS
                        && hint.at(husky::constants::kConsistency) == husky::constants::kBSP) {
                    int num_workers = stoi(hint.at(husky::constants::kNumWorkers));
                    if (hint.find(husky::constants::kUpdateType) != hint.end()
                            || hint.at(husky::constants::kUpdateType) == husky::constants::kAddUpdate) {
                        // bsp unordered_map, add
                        server_base_.reset(new BSPServer<Val, std::unordered_map<husky::constants::Key, Val>>(server_id, num_workers, std::move(store), false, false));
                    } else {
                        // bsp unordered_map, assign
                        server_base_.reset(new BSPServer<Val, std::unordered_map<husky::constants::Key, Val>>(server_id, num_workers, std::move(store), false, false));
                    }
                } else if (hint.at(husky::constants::kType) == husky::constants::kPS
                        && hint.at(husky::constants::kConsistency) == husky::constants::kSSP) {
                    int num_workers = stoi(hint.at(husky::constants::kNumWorkers));
                    int staleness = stoi(hint.at(husky::constants::kStaleness));
                    if (hint.find(husky::constants::kUpdateType) != hint.end()
                            || hint.at(husky::constants::kUpdateType) == husky::constants::kAddUpdate) {    // ssp unordered_map, add 
                        server_base_.reset(new SSPServer<Val, std::unordered_map<husky::constants::Key, Val>>(server_id, num_workers, std::move(store), false, false, staleness));
                    } else {   // ssp unordered_map, assign
                        server_base_.reset(new SSPServer<Val, std::unordered_map<husky::constants::Key, Val>>(server_id, num_workers, std::move(store), false, true, staleness));
                    }
                } else {
                    throw husky::base::HuskyException("Unknown hint");
                }
            } else if (hint.at(husky::constants::kStorageType) == husky::constants::kVectorStorage) {  
                std::vector<Val> store;
                store.resize(RangeManager::Get().GetServerSize(kv_id, server_id));
                if (hint.find(husky::constants::kType) == hint.end()
                        || hint.at(husky::constants::kType) == husky::constants::kSingle
                        || hint.at(husky::constants::kType) == husky::constants::kHogwild
                        || hint.at(husky::constants::kType) == husky::constants::kSPMT) {  // default is assign
                    server_base_.reset(new DefaultUpdateServer<Val,
                        std::vector<Val>>(kv_id, server_id, std::move(store), true, true));  // vector, assign
                } else if (hint.at(husky::constants::kType) == husky::constants::kPS
                        && hint.at(husky::constants::kConsistency) == husky::constants::kASP) { // vector, add
                    server_base_.reset(new DefaultUpdateServer<Val,
                        std::vector<Val>>(kv_id, server_id, std::move(store), true, false));
                } else if (hint.at(husky::constants::kType) == husky::constants::kPS
                        && hint.at(husky::constants::kConsistency) == husky::constants::kBSP) {
                    int num_workers = stoi(hint.at(husky::constants::kNumWorkers));    
                    if (hint.find(husky::constants::kUpdateType) != hint.end()
                            || hint.at(husky::constants::kUpdateType) == husky::constants::kAddUpdate) {
                        // bsp vector add
                        server_base_.reset(new BSPServer<Val, std::vector<Val>>(server_id, num_workers, std::move(store), true, false));
                    } else {
                        // bsp vector assign
                        server_base_.reset(new BSPServer<Val, std::vector<Val>>(server_id, num_workers, std::move(store), true, true));
                    }
                } else if (hint.at(husky::constants::kType) == husky::constants::kPS
                        && hint.at(husky::constants::kConsistency) == husky::constants::kSSP) {
                    int num_workers = stoi(hint.at(husky::constants::kNumWorkers));
                    int staleness = stoi(hint.at(husky::constants::kStaleness));
                    if (hint.find(husky::constants::kUpdateType) != hint.end()
                            || hint.at(husky::constants::kUpdateType) == husky::constants::kAddUpdate) {
                        // ssp vector, add
                        server_base_.reset(new SSPServer<Val, std::vector<Val>>(server_id, num_workers, std::move(store), true, false, staleness));
                    } else {
                        // ssp vector, assign
                        server_base_.reset(new SSPServer<Val, std::vector<Val>>(server_id, num_workers, std::move(store), true, true, staleness));
                    }
                } else {
                    throw husky::base::HuskyException("Unknown hint");
                }
            } else {
                throw husky::base::HuskyException("Unknown hint");
            }
        } catch (...) {
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
