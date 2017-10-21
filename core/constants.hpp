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

// hdfs_block
const uint32_t kIOHDFSSubsetLoad = 301;
// hdfs_binary
const uint32_t kIOHDFSBinarySubsetLoad = 302;

/*
 * TODO: This is only used in Load and Dump. May put in a better place
 */
constexpr const char* const kKVStoreChunks = "kvstore_chunks";
constexpr const char* const kKVStoreIntegral = "kvstore_integral";
constexpr const char* const kTransferIntegral = "transfer_integral";

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
