include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/husky ${Boost_INCLUDE_DIRS})

if(LIBHDFS3_FOUND)
    include_directories(${LIBHDFS3_INCLUDE_DIR})
    add_definitions(-DWITH_HDFS)
endif(LIBHDFS3_FOUND)

if(MONGOCLIENT_FOUND)
    include_directories(${MONGOCLIENT_INCLUDE_DIR})
    add_definitions(-DWITH_MONGODB)
endif(MONGOCLIENT_FOUND)

# Unit Test
file(GLOB_RECURSE UNITTEST_FILES ${PROJECT_SOURCE_DIR}/*_ml_unittest.cpp) 
remove_definitions(-DWITH_THRIFT)  # Workaround for namespace collisions between gtest and flume.
include_directories(${GTEST_INCLUDE})
add_executable(HuskyMLUnitTest ${UNITTEST_FILES} unittest_main.cpp)
add_dependencies(HuskyMLUnitTest gtest)
target_link_libraries(HuskyMLUnitTest husky)
target_link_libraries(HuskyMLUnitTest ${HUSKY_EXTERNAL_LIB})
# hard code GTEST_LIBRARIES and GMOCK_LIBRARIES
set(GTEST_LIBRARIES_TMP
    "${PROJECT_BINARY_DIR}/husky/lib/libgtest.a"
    "${PROJECT_BINARY_DIR}/husky/lib/libgtest_main.a")
set(GMOCK_LIBRARIES_TMP
    "${PROJECT_BINARY_DIR}/husky/lib/libgmock.a"
    "${PROJECT_BINARY_DIR}/husky/lib/libgmock_main.a")
target_link_libraries(HuskyMLUnitTest ${GTEST_LIBRARIES_TMP})
target_link_libraries(HuskyMLUnitTest ${GMOCK_LIBRARIES_TMP})
husky_default_properties(HuskyMLUnitTest)

# TestPI
add_executable(TestPI test_pi.cpp)
target_link_libraries(TestPI husky)
target_link_libraries(TestPI ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestPI PROPERTY CXX_STANDARD 14)

# TestDatastore
add_executable(TestDatastore test_datastore.cpp)
target_link_libraries(TestDatastore husky)
target_link_libraries(TestDatastore ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestDatastore PROPERTY CXX_STANDARD 14)

# TestLoad
add_executable(TestLoad test_load.cpp)
target_link_libraries(TestLoad husky)
target_link_libraries(TestLoad ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestLoad PROPERTY CXX_STANDARD 14)

# TestLoadData
add_executable(TestLoadData test_load_data.cpp)
target_link_libraries(TestLoadData husky)
target_link_libraries(TestLoadData ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestLoadData PROPERTY CXX_STANDARD 14)

# TestGreedy
add_executable(TestGreedy test_greedy_task_scheduler.cpp)
target_link_libraries(TestGreedy husky)
target_link_libraries(TestGreedy ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestGreedy PROPERTY CXX_STANDARD 14)

# TestMigrate
add_executable(TestMigrate test_migrate.cpp)
target_link_libraries(TestMigrate husky)
target_link_libraries(TestMigrate ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestMigrate PROPERTY CXX_STANDARD 14)

# TestGlobalize
add_executable(TestGlobalize test_globalize.cpp)
target_link_libraries(TestGlobalize husky)
target_link_libraries(TestGlobalize ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestGlobalize PROPERTY CXX_STANDARD 14)

# TestSVRG
add_executable(TestSVRG test_svrg.cpp)
target_link_libraries(TestSVRG husky)
target_link_libraries(TestSVRG ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestSVRG PROPERTY CXX_STANDARD 14)

# TestSampleReader
add_executable(TestSampleReader test_sample_reader.cpp)
target_link_libraries(TestSampleReader husky)
target_link_libraries(TestSampleReader ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestSampleReader PROPERTY CXX_STANDARD 14)

# TestConfigurableTask
add_executable(TestConfigurableTask test_configurableTask.cpp)
target_link_libraries(TestConfigurableTask husky)
target_link_libraries(TestConfigurableTask ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestConfigurableTask PROPERTY CXX_STANDARD 14)

# TestDataSampler
add_executable(TestDataSampler test_data_sampler.cpp)
target_link_libraries(TestDataSampler husky)
target_link_libraries(TestDataSampler ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestDataSampler PROPERTY CXX_STANDARD 14)

# TestModel
add_executable(TestModel test_model.cpp)
target_link_libraries(TestModel husky)
target_link_libraries(TestModel ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestModel PROPERTY CXX_STANDARD 14)

# TestLoadDump
add_executable(TestLoadDump test_load_dump.cpp)
target_link_libraries(TestLoadDump husky)
target_link_libraries(TestLoadDump ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestLoadDump PROPERTY CXX_STANDARD 14)
