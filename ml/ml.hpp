#pragma once

#include <memory>

#include "core/table_info.hpp"

#include "husky/base/exception.hpp"
#include "husky/core/context.hpp"
#include "core/info.hpp"
#include "core/utility.hpp"

#include "ml/mlworker/mlworker.hpp"
#include "ml/mlworker/hogwild.hpp"
#include "ml/mlworker/spmt.hpp"
#include "ml/mlworker/ps.hpp"
#include "ml/mlworker/single.hpp"

namespace ml {

template<typename Val>
std::unique_ptr<mlworker::GenericMLWorker<Val>> 
CreateMLWorker(const husky::Info& info, const husky::TableInfo& table_info) {
    std::unique_ptr<mlworker::GenericMLWorker<Val>> mlworker;
    if (table_info.mode_type == husky::ModeType::Single) {
        mlworker.reset(new ml::mlworker::SingleWorker<Val>(info, table_info));
    } else if (table_info.mode_type == husky::ModeType::Hogwild) {
        mlworker.reset(new ml::mlworker::HogwildWorker<Val>(info, table_info,
            *husky::Context::get_zmq_context()));
    } else if (table_info.mode_type == husky::ModeType::SPMT) {
        mlworker.reset(new ml::mlworker::SPMTWorker<Val>(info, table_info,
            *husky::Context::get_zmq_context()));
    } else if (table_info.mode_type == husky::ModeType::PS) {
        husky::LOG_I << "using " << husky::WorkerTypeName[static_cast<int>(table_info.worker_type)];
        if (table_info.worker_type == husky::WorkerType::PSWorker) {
            mlworker.reset(new ml::mlworker::PSWorker<Val>(info, table_info));
        } else if (table_info.worker_type == husky::WorkerType::PSMapNoneWorker) {
            mlworker.reset(new ml::mlworker::PSMapNoneWorker<Val>(info, table_info));
        } else if (table_info.worker_type == husky::WorkerType::PSChunkChunkWorker) {
            mlworker.reset(new ml::mlworker::PSChunkChunkWorker<Val>(info, table_info, *husky::Context::get_zmq_context()));
        } else if (table_info.worker_type == husky::WorkerType::PSChunkNoneWorker) {
            mlworker.reset(new ml::mlworker::PSChunkNoneWorker<Val>(info, table_info));
        } else if (table_info.worker_type == husky::WorkerType::PSMapChunkWorker) {
            mlworker.reset(new ml::mlworker::PSMapChunkWorker<Val>(info, table_info, *husky::Context::get_zmq_context()));
        } else if (table_info.worker_type == husky::WorkerType::PSNoneChunkWorker) {
            mlworker.reset(new ml::mlworker::PSNoneChunkWorker<Val>(info, table_info, *husky::Context::get_zmq_context()));
        } else if (table_info.worker_type == husky::WorkerType::PSBspWorker) {
            mlworker.reset(new ml::mlworker::PSBspWorker<Val>(info, table_info, *husky::Context::get_zmq_context()));
        } else {
            husky::LOG_I << "table_info error: " << table_info.DebugString();
            assert(false);
        }
    } else {
        husky::LOG_I << "table_info error: " << table_info.DebugString();
        assert(false);
    }
    return mlworker;
}

}  // ml
