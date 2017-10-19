#pragma once

#include <memory>

#include "husky/core/context.hpp"
#include "worker/basic.hpp"
#include "worker/model_transfer_manager.hpp"
#include "worker/task_factory.hpp"
#include "worker/worker.hpp"

#include "husky/core/constants.hpp"
#include "husky/core/coordinator.hpp"
#include "husky/core/job_runner.hpp"

namespace husky {

/*
 * Engine manages the process
 */
class Engine {
   public:
    static Engine& Get() {
        static Engine engine;
        return engine;
    }
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    /*
     * Add a new task to the buffer
     */
    template <typename TaskT>
    void AddTask(const TaskT& task, const FuncT& func) {
        worker->add_task(task, func);
    }

    /*
     * Submit the buffered tasks to cluster_manager
     *
     * It's a blocking method, return when all the buffered tasks are finished
     */
    void Submit();
    /*
     * Ask the ClusterManager to exit
     *
     * It means that no more tasks will submit. Basically the end of the process
     */
    void Exit();

   private:
    // The constructor
    Engine();

    /*
     * Start function to initialize the environment
     */
    void StartWorker();

    void StartCoordinator();

    // Function to stop the worker
    void StopWorker();

    // Function to stop the coordinator
    void StopCoordinator();

    std::unique_ptr<Worker> worker;
    std::unique_ptr<ModelTransferManager> model_transfer_manager;
};

}  // namespace husky
