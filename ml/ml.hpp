#pragma once

#include <memory>

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
CreateMLWorker(const husky::Info& info) {
    auto& hint = info.get_hint();
    std::unique_ptr<mlworker::GenericMLWorker<Val>> mlworker;
    try {
        if (hint.at(husky::constants::kType) == husky::constants::kPS) {
            if (hint.find(husky::constants::kWorkerType) != hint.end()) {
                if (hint.at(husky::constants::kWorkerType) == husky::constants::kPSWorker) {
                    husky::LOG_I << "using PSWorker";
                    mlworker.reset(new ml::mlworker::PSWorker<Val>(info));
                } else if (hint.at(husky::constants::kWorkerType) == husky::constants::kSSPWorker) {
                    husky::LOG_I << "using SSPWorker";
                    mlworker.reset(new ml::mlworker::SSPWorker<Val>(info));
                } else if (hint.at(husky::constants::kWorkerType) == husky::constants::kPSSharedWorkerChunk) {
                    husky::LOG_I << "using PSSharedChunkWorker";
                    mlworker.reset(new ml::mlworker::PSSharedChunkWorker<Val>(info, *husky::Context::get_zmq_context()));
                } else if (hint.at(husky::constants::kWorkerType) == husky::constants::kSSPWorkerChunk) {
                    husky::LOG_I << "using SSPWorkerChunk";
                    mlworker.reset(new ml::mlworker::SSPWorkerChunk<Val>(info));
                } else if (hint.at(husky::constants::kWorkerType) == husky::constants::kPSSharedWorker) {
                    husky::LOG_I << "using PSSharedWorker";
                    mlworker.reset(new ml::mlworker::PSSharedWorker<Val>(info, *husky::Context::get_zmq_context()));
                } 
            } else {
                mlworker.reset(new ml::mlworker::PSWorker<Val>(info));
            }
        } else if (hint.at(husky::constants::kType) == husky::constants::kSingle) {
            mlworker.reset(new ml::mlworker::SingleWorker<Val>(info));
        } else if (hint.at(husky::constants::kType) == husky::constants::kHogwild) {
            mlworker.reset(new ml::mlworker::HogwildWorker<Val>(info,
                *husky::Context::get_zmq_context()));
        } else if (hint.at(husky::constants::kType) == husky::constants::kSPMT) {
            mlworker.reset(new ml::mlworker::SPMTWorker<Val>(info,
                *husky::Context::get_zmq_context()));
        } else {
            throw;
        }
    } catch(...) {
        husky::utility::print_hint(hint);
        throw husky::base::HuskyException("ml.hpp: Unknown hint");
    }
    return mlworker;
}

}  // ml
