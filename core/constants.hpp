#pragma once 

#include <cstdint> 

namespace husky {
namespace constants {

namespace {

using Key = uint64_t;

// TODO: magic number: channel id reserved for kvstore
const int kv_channel_id = 37;

const uint32_t kClusterManagerInit = 200;
const uint32_t kClusterManagerThreadFinished = 201;
const uint32_t kClusterManagerDirectTransferModel = 202;
const uint32_t kClusterManagerExit = 203;
const uint32_t kClusterManagerTimeOutSchedule = 204;

const uint32_t kTaskType = 100;
const uint32_t kThreadFinished = 101;
const uint32_t kClusterManagerFinished = 102;

const uint32_t kIOHDFSSubsetLoad = 301;

/*
 * storage type
 *
 * If Users want to use kvstore directly, they may set the kStorageType explicitly to
 * control the storage format of the kvstore.
 *
 * The default storage type is kUnorderedMapStorage. 
 */
constexpr const char* const kStorageType = "storage_type";
constexpr const char* const kVectorStorage = "vector_storage";
constexpr const char* const kUnorderedMapStorage = "unordered_map_storage";

/*
 * kvstore update type
 *
 * If Users want to use kvstore directly, they may set the kUpdateType explicitly to 
 * control the update methods of kvstore
 *
 * add_update means store[key] += val
 * assign_update means store[key] = val 
 */
constexpr const char* const kUpdateType = "update_type";
constexpr const char* const kAddUpdate = "add_update";
constexpr const char* const kAssignUpdate = "assign_update";
/*
 * type
 *
 * ML user needs set the worker type explicitly.
 */
constexpr const char* const kType = "type";
constexpr const char* const kSingle = "Single";
constexpr const char* const kHogwild = "Hogwild";
constexpr const char* const kSPMT = "SPMT";
constexpr const char* const kPS = "PS";

/*
 * consistency
 *
 * The consistency level can be kSSP, kBSP, kASP.
 * User using PS and SPMT may need to set kConsistency
 */
constexpr const char* const kConsistency = "consistency";
constexpr const char* const kSSP = "SSP";
constexpr const char* const kBSP = "BSP";
constexpr const char* const kASP = "ASP";

// worker type
constexpr const char* const kWorkerType = "worker_type";
constexpr const char* const kPSWorker = "PSWorker";
constexpr const char* const kSSPWorker = "SSPWorker";
constexpr const char* const kSSPWorkerChunk = "SSPWorkerChunk";
constexpr const char* const kPSSharedWorker = "PSSharedWorker";
constexpr const char* const kPSSharedWorkerChunk = "PSSharedChunkWorker";

constexpr const char* const kNumWorkers = "num_workers";
constexpr const char* const kStaleness = "staleness";


constexpr const char* const kEnableDirectModelTransfer = "direct_model_transfer";
constexpr const char* const kKVStoreChunks = "kvstore_chunks";
constexpr const char* const kKVStoreIntegral = "kvstore_integral";
constexpr const char* const kTransferIntegral = "transfer_integral";

/*
 * param_type
 */
constexpr const char* const kParamType = "param_type";
constexpr const char* const kIntegralType = "integral_type";
constexpr const char* const kChunkType = "chunk_type";

/*
 * load_type 
 *
 * load_hdfs_locally means the thread only can access the data in its process
 * load_hdfs_globally means when the thread loads all the data int its process, it can access the data in other processes  
 */
constexpr const char* const kLoadHdfsType = "load_hdfs_type";
constexpr const char* const kLoadHdfsLocally = "load_hdfs_locally";
constexpr const char* const kLoadHdfsGlobally = "load_hdfs_globally";

}  // namespace anonymous
}  // namespace constants
}  // namespace husky
