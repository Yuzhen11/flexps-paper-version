# Copyright 2016 Husky Team
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/husky ${Boost_INCLUDE_DIRS})

if(LIBHDFS3_FOUND)
    include_directories(${LIBHDFS3_INCLUDE_DIR})
    add_definitions(-DWITH_HDFS)
endif(LIBHDFS3_FOUND)

if(MONGOCLIENT_FOUND)
    include_directories(${MONGOCLIENT_INCLUDE_DIR})
    add_definitions(-DWITH_MONGODB)
endif(MONGOCLIENT_FOUND)

# set(husky husky-lib husky-io husky-core husky-base) 
# set(proj core base husky) 


### Applications

# ProcMain
add_executable(ProcMain ProcMain.cpp)
target_link_libraries(ProcMain husky)
target_link_libraries(ProcMain ${HUSKY_EXTERNAL_LIB})

set_property(TARGET ProcMain PROPERTY CXX_STANDARD 14)

# ClusterManagerMain
add_executable(ClusterManagerMain ClusterManagerMain.cpp)
target_link_libraries(ClusterManagerMain husky)
target_link_libraries(ClusterManagerMain ${HUSKY_EXTERNAL_LIB})
set_property(TARGET ClusterManagerMain PROPERTY CXX_STANDARD 14)

# ProcMainWithContext
add_executable(ProcMainWithContext ProcMainWithContext.cpp)
target_link_libraries(ProcMainWithContext husky)
target_link_libraries(ProcMainWithContext ${HUSKY_EXTERNAL_LIB})
set_property(TARGET ProcMainWithContext PROPERTY CXX_STANDARD 14)

# ProcMainWithContextDriver
add_executable(ProcMainWithContextDriver ProcMainWithContextDriver.cpp)
target_link_libraries(ProcMainWithContextDriver husky)
target_link_libraries(ProcMainWithContextDriver ${HUSKY_EXTERNAL_LIB})
set_property(TARGET ProcMainWithContextDriver PROPERTY CXX_STANDARD 14)

# ClusterManagerMainWithContext
add_executable(ClusterManagerMainWithContext ClusterManagerMainWithContext.cpp $<TARGET_OBJECTS:husky-master-objs>)
target_link_libraries(ClusterManagerMainWithContext husky)
target_link_libraries(ClusterManagerMainWithContext ${HUSKY_EXTERNAL_LIB})
set_property(TARGET ClusterManagerMainWithContext PROPERTY CXX_STANDARD 14)

# TestPI
add_executable(TestPI pi.cpp)
target_link_libraries(TestPI husky)
target_link_libraries(TestPI ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestPI PROPERTY CXX_STANDARD 14)

# SimpleKVApp
add_executable(SimpleKVApp simple_kv_app.cpp)
target_link_libraries(SimpleKVApp husky)
target_link_libraries(SimpleKVApp ${HUSKY_EXTERNAL_LIB})
set_property(TARGET SimpleKVApp PROPERTY CXX_STANDARD 14)

# TestKVApp
add_executable(TestKVApp test_kv_app.cpp)
target_link_libraries(TestKVApp husky)
target_link_libraries(TestKVApp ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestKVApp PROPERTY CXX_STANDARD 14)

# SimpleHogwild
add_executable(SimpleHogwild simple_hogwild.cpp)
target_link_libraries(SimpleHogwild husky)
target_link_libraries(SimpleHogwild ${HUSKY_EXTERNAL_LIB})
set_property(TARGET SimpleHogwild PROPERTY CXX_STANDARD 14)

# SimpleKVStore
add_executable(SimpleKVStore simple_kvstore.cpp)
target_link_libraries(SimpleKVStore husky)
target_link_libraries(SimpleKVStore ${HUSKY_EXTERNAL_LIB})
set_property(TARGET SimpleKVStore PROPERTY CXX_STANDARD 14)

# GenericML
add_executable(GenericML generic_ml.cpp)
target_link_libraries(GenericML husky)
target_link_libraries(GenericML ${HUSKY_EXTERNAL_LIB})
set_property(TARGET GenericML PROPERTY CXX_STANDARD 14)

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

# LogisticRegression
add_executable(SingleLR logistic_regression.cpp)
target_link_libraries(SingleLR husky)
target_link_libraries(SingleLR ${HUSKY_EXTERNAL_LIB})
set_property(TARGET SingleLR PROPERTY CXX_STANDARD 14)

# LogisticRegression
add_executable(SingleLRGeneric logistic_regression_generic.cpp)
target_link_libraries(SingleLRGeneric husky)
target_link_libraries(SingleLRGeneric ${HUSKY_EXTERNAL_LIB})
set_property(TARGET SingleLRGeneric PROPERTY CXX_STANDARD 14)

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

# TestFixedWorkers
add_executable(TestFixedWorkers test_fixedworkers.cpp)
target_link_libraries(TestFixedWorkers husky)
target_link_libraries(TestFixedWorkers ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestFixedWorkers PROPERTY CXX_STANDARD 14)

# TestSampleReader
add_executable(TestSampleReader sample_reader.cpp)
target_link_libraries(TestSampleReader husky)
target_link_libraries(TestSampleReader ${HUSKY_EXTERNAL_LIB})
set_property(TARGET TestSampleReader PROPERTY CXX_STANDARD 14)

# LRSampleReader
add_executable(LRSampleReader logistic_regression_samplereader.cpp)
target_link_libraries(LRSampleReader husky)
target_link_libraries(LRSampleReader ${HUSKY_EXTERNAL_LIB})
set_property(TARGET LRSampleReader PROPERTY CXX_STANDARD 14)
