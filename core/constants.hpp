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

// Communication type
const uint32_t COMM_PUSH = 0;
const uint32_t COMM_REQ = 1;
const uint32_t COMM_REP = 2;
const uint32_t COMM_MIGR = 3;
const uint32_t COMM_END = 4;
const uint32_t COMM_PRGS = 5;
const uint32_t COMM_PROBE = 6;

/// Mailbox
const uint32_t MAILBOX_EVENT_RECV_COMM = 0x2f3b1a66;
const uint32_t MAILBOX_EVENT_RECV_COMM_PRGS = 0x303b1366;
const uint32_t MAILBOX_EVENT_RECV_COMM_END = 0x30e31266;
const uint32_t MAILBOX_EVENT_RECV_COMM_PUSH = (COMM_PUSH << 2);
const uint32_t MAILBOX_EVENT_RECV_COMM_MIGR = (COMM_MIGR << 2);
const uint32_t MAILBOX_EVENT_RECV_COMM_REP = (COMM_REP << 2);
const uint32_t MAILBOX_EVENT_RECV_COMM_REQ = (COMM_REQ << 2);
const uint32_t MAILBOX_EVENT_SEND_COMM = 0x30eb1266;
const uint32_t MAILBOX_EVENT_SEND_COMM_PRGS = 0x33eb1266;
const uint32_t MAILBOX_EVENT_SEND_COMM_END = 0x303b1266;
const uint32_t MAILBOX_EVENT_DESTROY = 0x303b1276;


}  // namespace constants
}  // namespace husky
