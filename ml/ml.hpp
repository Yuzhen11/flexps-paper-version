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
                } else if (hint.at(husky::constants::kWorkerType) == husky::constants::kPSMapNoneWorker) {
                    husky::LOG_I << "using PSMapNoneWorker";
                    mlworker.reset(new ml::mlworker::PSMapNoneWorker<Val>(info));
                } else if (hint.at(husky::constants::kWorkerType) == husky::constants::kPSChunkChunkWorker) {
                    husky::LOG_I << "using PSChunkChunkWorker";
                    mlworker.reset(new ml::mlworker::PSChunkChunkWorker<Val>(info, *husky::Context::get_zmq_context()));
                } else if (hint.at(husky::constants::kWorkerType) == husky::constants::kPSChunkNoneWorker) {
                    husky::LOG_I << "using PSChunkNoneWorker";
                    mlworker.reset(new ml::mlworker::PSChunkNoneWorker<Val>(info));
                } else if (hint.at(husky::constants::kWorkerType) == husky::constants::kPSMapChunkWorker) {
                    husky::LOG_I << "using PSMapChunkWorker";
                    mlworker.reset(new ml::mlworker::PSMapChunkWorker<Val>(info, *husky::Context::get_zmq_context()));
                } else if (hint.at(husky::constants::kWorkerType) == husky::constants::kPSNoneChunkWorker) {
                    husky::LOG_I << "using PSNoneChunkWorker";
                    mlworker.reset(new ml::mlworker::PSNoneChunkWorker<Val>(info, *husky::Context::get_zmq_context()));
                } else if (hint.at(husky::constants::kWorkerType) == husky::constants::kPSBspWorker) {
                    husky::LOG_I << "using PSBspWorker";
                    mlworker.reset(new ml::mlworker::PSBspWorker<Val>(info, *husky::Context::get_zmq_context()));
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
