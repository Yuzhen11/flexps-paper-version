#pragma once

namespace husky {
namespace constants {

// TODO: magic number: channel id reserved for kvstore
const int kv_channel_id = 37;

const uint32_t MASTER_INIT = 200;
const uint32_t MASTER_INSTANCE_FINISHED = 201;
const uint32_t MASTER_EXIT = 202;


const uint32_t TASK_TYPE = 100;
const uint32_t THREAD_FINISHED = 101;
const uint32_t MASTER_FINISHED = 102;



}  // namespace constants
}  // namespace husky
