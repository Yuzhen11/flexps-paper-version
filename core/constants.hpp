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

const uint32_t kTaskType = 100;
const uint32_t kThreadFinished = 101;
const uint32_t kClusterManagerFinished = 102;

const uint32_t kIOHDFSSubsetLoad = 301;

// storage type
constexpr const char* const kStorageType = "storage_type";
constexpr const char* const kVectorStorage = "vector_storage";
constexpr const char* const kUnorderedMapStorage = "unordered_storage";

// type
constexpr const char* const kType = "type";
constexpr const char* const kSingle = "Single";
constexpr const char* const kHogwild = "Hogwild";
constexpr const char* const kSPMT = "SPMT";
constexpr const char* const kPS = "PS";

// consistency
constexpr const char* const kConsistency = "consistency";
constexpr const char* const kSSP = "SSP";
constexpr const char* const kBSP = "BSP";
constexpr const char* const kASP = "ASP";

// worker type
constexpr const char* const kWorkerType = "worker_type";
constexpr const char* const kSSPWorker = "SSPWorker";

constexpr const char* const kNumWorkers = "num_workers";
constexpr const char* const kStaleness = "staleness";


}  // namespace anonymous

}  // namespace constants
}  // namespace husky
