#pragma once

#include <sstream>

namespace husky {

enum class ModeType {
    Single, Hogwild, SPMT, PS,
    None
};
static const char* ModeTypeName[] = {
    "Single", "Hogwild", "SPMT", "PS",
    "None"
};

enum class Consistency {
    SSP, BSP, ASP,
    None
};
static const char* ConsistencyName[] = {
    "SSP", "BSP", "ASP",
    "None"
};

enum class WorkerType {
    PSWorker, PSMapNoneWorker, PSChunkNoneWorker, PSMapChunkWorker, PSChunkChunkWorker,
    PSNoneChunkWorker, PSBspWorker,
    None,
};
static const char* WorkerTypeName[] = {
    "PSWorker", "PSMapNoneWorker", "PSChunkNoneWorker", "PSMapChunkWorker", "PSChunkChunkWorker",
    "PSNoneChunkWorker", "PSBspWorker",
    "None",
};

enum class ParamType {
    IntegralType, ChunkType,
    None
};
static const char* ParamTypeName[] = {
    "IntegralType", "ChunkType",
    "None"
};

enum class CacheStrategy {
    LRU, LFU, Random,
    None
};
static const char* CacheStrategyName[] = {
    "LRU", "LFU", "Random",
    "None"
};

/*
 * TODO
constexpr const char* const kKVStoreChunks = "kvstore_chunks";
constexpr const char* const kKVStoreIntegral = "kvstore_integral";
constexpr const char* const kTransferIntegral = "transfer_integral";
*/

struct TableInfo {
    struct CacheInfo {
        const CacheStrategy cache_strategy = CacheStrategy::None;
        const int threshold;
        const float dump_factor;
        std::string DebugString() const {
            std::stringstream ss;
            ss << "{";
            ss << " CacheStrategy:" << CacheStrategyName[static_cast<int>(cache_strategy)];
            ss << " threshold:" << threshold;
            ss << " dump_factor:" << dump_factor;
            ss << "}";
            return ss.str();
        }
    };
    const int kv_id;
    const int dims;
    const ModeType mode_type = ModeType::None;
    Consistency consistency = Consistency::None;
    const WorkerType worker_type = WorkerType::None;
    ParamType param_type = ParamType::None;
    int kStaleness;
    const bool kEnableDirectModelTransfer;
    const CacheInfo cache_info;

    std::string DebugString() const {
        std::stringstream ss;
        ss << "{";
        ss << " id:" << kv_id;
        ss << " dims:" << dims;
        ss << " ModeType:" << ModeTypeName[static_cast<int>(mode_type)];
        ss << " Consistency:" << ConsistencyName[static_cast<int>(consistency)];
        ss << " WorkerType:" << WorkerTypeName[static_cast<int>(worker_type)];
        ss << " ParamType:" << ParamTypeName[static_cast<int>(param_type)];
        ss << " kStaleness:" << kStaleness;
        ss << " kEnableDirectModelTransfer:" << kEnableDirectModelTransfer;
        ss << " " << cache_info.DebugString();
        ss << "}";
        return ss.str();
    }
};

}  // namespace husky
