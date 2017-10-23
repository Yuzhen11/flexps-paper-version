# ALS
add_executable(ALS als.cpp)
target_link_libraries(ALS husky)
target_link_libraries(ALS husky-ml)
if(USE_PROFILER)
target_link_libraries(ALS profiler)
endif(USE_PROFILER)
target_link_libraries(ALS ${HUSKY_EXTERNAL_LIB})
husky_default_properties(ALS)

# ALSshared
add_executable(ALSshared als_shared.cpp)
target_link_libraries(ALSshared husky)
target_link_libraries(ALSshared husky-ml)
if(USE_PROFILER)
target_link_libraries(ALSshared profiler)
endif(USE_PROFILER)
target_link_libraries(ALSshared ${HUSKY_EXTERNAL_LIB})
husky_default_properties(ALSshared)
