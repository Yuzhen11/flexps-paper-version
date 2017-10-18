include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/husky ${HUSKY_EXTERNAL_INCLUDE})

file(GLOB mlmanager-scheduler-files
    task_manager.cpp
    history_manager.cpp
    priority_task_scheduler.cpp
    sequential_task_scheduler.cpp
    greedy_task_scheduler.cpp
    auto_parallelism_task_scheduler.cpp
    available_workers.cpp
    task_scheduler_utils.cpp)
husky_cache_variable(mlmanager-scheduler-files ${mlmanager-scheduler-files})

add_library(mlmanager-scheduler-objs OBJECT ${mlmanager-scheduler-files})
husky_default_properties(mlmanager-scheduler-objs)
