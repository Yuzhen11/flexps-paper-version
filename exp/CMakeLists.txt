include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/husky ${HUSKY_EXTERNAL_INCLUDE})

# ReadLocalGlobal
add_executable(ReadLocalGlobal read_local_global.cpp)
target_link_libraries(ReadLocalGlobal husky)
target_link_libraries(ReadLocalGlobal husky-ml)
target_link_libraries(ReadLocalGlobal ${HUSKY_EXTERNAL_LIB})
husky_default_properties(ReadLocalGlobal)
