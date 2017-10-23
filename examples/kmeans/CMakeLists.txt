# Kmeans
add_executable(Kmeans kmeans.cpp)
target_link_libraries(Kmeans husky)
target_link_libraries(Kmeans husky-ml)
if(USE_PROFILER)
target_link_libraries(Kmeans profiler)
endif(USE_PROFILER)
target_link_libraries(Kmeans ${HUSKY_EXTERNAL_LIB})
husky_default_properties(Kmeans)

# Kmeans_multi_stage
add_executable(Kmeans_multi_stage kmeans_multi_stage.cpp)
target_link_libraries(Kmeans_multi_stage husky)
target_link_libraries(Kmeans_multi_stage husky-ml)
if(USE_PROFILER)
target_link_libraries(Kmeans_multi_stage profiler)
endif(USE_PROFILER)
target_link_libraries(Kmeans_multi_stage ${HUSKY_EXTERNAL_LIB})
husky_default_properties(Kmeans_multi_stage)
