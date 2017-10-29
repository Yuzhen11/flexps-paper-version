# MF_SGD
add_executable(MFSGD mf_sgd.cpp)
target_link_libraries(MFSGD husky)
target_link_libraries(MFSGD husky-ml)
if(USE_PROFILER)
target_link_libraries(MFSGD profiler)
endif(USE_PROFILER)
target_link_libraries(MFSGD ${HUSKY_EXTERNAL_LIB})
husky_default_properties(MFSGD)

