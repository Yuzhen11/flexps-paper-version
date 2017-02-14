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

# LogisticRegression
add_executable(SingleLR logistic_regression.cpp)
target_link_libraries(SingleLR husky)
target_link_libraries(SingleLR ${HUSKY_EXTERNAL_LIB})
set_property(TARGET SingleLR PROPERTY CXX_STANDARD 14)

# LogisticRegression Generic
add_executable(SingleLRGeneric logistic_regression_generic.cpp)
target_link_libraries(SingleLRGeneric husky)
target_link_libraries(SingleLRGeneric ${HUSKY_EXTERNAL_LIB})
set_property(TARGET SingleLRGeneric PROPERTY CXX_STANDARD 14)

# LRSampleReader
add_executable(LRSampleReader logistic_regression_samplereader.cpp)
target_link_libraries(LRSampleReader husky)
target_link_libraries(LRSampleReader ${HUSKY_EXTERNAL_LIB})
set_property(TARGET LRSampleReader PROPERTY CXX_STANDARD 14)

