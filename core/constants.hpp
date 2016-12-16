#pragma once

#include <cstdint>

namespace husky {
namespace constants {

namespace {

// TODO: magic number: channel id reserved for kvstore
const int kv_channel_id = 37;

const uint32_t kClusterManagerInit = 200;
const uint32_t kClusterManagerInstanceFinished = 201;
const uint32_t kClusterManagerExit = 202;

const uint32_t kTaskType = 100;
const uint32_t kThreadFinished = 101;
const uint32_t kClusterManagerFinished = 102;

}  // namespace anonymous

}  // namespace constants
}  // namespace husky
