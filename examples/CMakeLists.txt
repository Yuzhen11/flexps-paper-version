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


include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/husky ${HUSKY_EXTERNAL_INCLUDE})

add_subdirectory(lda)

### Applications

# ClusterManagerMain
add_executable(ClusterManagerMain ClusterManagerMain.cpp)
target_link_libraries(ClusterManagerMain husky-ml-master)
target_link_libraries(ClusterManagerMain husky)
target_link_libraries(ClusterManagerMain ${HUSKY_EXTERNAL_LIB})
husky_default_properties(ClusterManagerMain)

# ProcMainWithContextDriver
add_executable(ProcMainWithContextDriver ProcMainWithContextDriver.cpp)
target_link_libraries(ProcMainWithContextDriver husky)
target_link_libraries(ProcMainWithContextDriver husky-ml)
target_link_libraries(ProcMainWithContextDriver ${HUSKY_EXTERNAL_LIB})
husky_default_properties(ProcMainWithContextDriver)

# ClusterManagerMainWithContext
add_executable(ClusterManagerMainWithContext ClusterManagerMainWithContext.cpp $<TARGET_OBJECTS:husky-master-objs>)
target_link_libraries(ClusterManagerMainWithContext husky)
target_link_libraries(ClusterManagerMainWithContext husky-ml-master)
target_link_libraries(ClusterManagerMainWithContext ${HUSKY_EXTERNAL_LIB})
husky_default_properties(ClusterManagerMainWithContext)

# SimpleKVStore
add_executable(SimpleKVStore simple_kvstore.cpp)
target_link_libraries(SimpleKVStore husky)
target_link_libraries(SimpleKVStore husky-ml)
target_link_libraries(SimpleKVStore ${HUSKY_EXTERNAL_LIB})
husky_default_properties(SimpleKVStore)

# GenericML
add_executable(GenericML generic_ml.cpp)
target_link_libraries(GenericML husky)
target_link_libraries(GenericML husky-ml)
target_link_libraries(GenericML ${HUSKY_EXTERNAL_LIB})
husky_default_properties(GenericML)

# LogisticRegression
add_executable(SingleLR logistic_regression.cpp)
target_link_libraries(SingleLR husky)
target_link_libraries(SingleLR husky-ml)
target_link_libraries(SingleLR ${HUSKY_EXTERNAL_LIB})
husky_default_properties(SingleLR)

# LogisticRegression Generic
add_executable(SingleLRGeneric logistic_regression_generic.cpp)
target_link_libraries(SingleLRGeneric husky)
target_link_libraries(SingleLRGeneric husky-ml)
target_link_libraries(SingleLRGeneric ${HUSKY_EXTERNAL_LIB})
husky_default_properties(SingleLRGeneric)

# MultiTasks
add_executable(MultiTasks multi_ml.cpp)
target_link_libraries(MultiTasks husky)
target_link_libraries(MultiTasks husky-ml)
target_link_libraries(MultiTasks ${HUSKY_EXTERNAL_LIB})
husky_default_properties(MultiTasks)

# LRSampleReader
add_executable(LRSampleReader logistic_regression_samplereader.cpp)
target_link_libraries(LRSampleReader husky)
target_link_libraries(LRSampleReader husky-ml)
target_link_libraries(LRSampleReader ${HUSKY_EXTERNAL_LIB})
husky_default_properties(LRSampleReader)

# LRSampleReaderParse
add_executable(LRSampleReaderParse logistic_regression_simplesr.cpp)
target_link_libraries(LRSampleReaderParse husky)
target_link_libraries(LRSampleReaderParse husky-ml)
target_link_libraries(LRSampleReaderParse ${HUSKY_EXTERNAL_LIB})
husky_default_properties(LRSampleReaderParse)

# SPMT_PS
add_executable(SPMT_PS spmt_ps_compare.cpp)
target_link_libraries(SPMT_PS husky)
target_link_libraries(SPMT_PS husky-ml)
target_link_libraries(SPMT_PS ${HUSKY_EXTERNAL_LIB})
husky_default_properties(SPMT_PS)

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

# Kmeans
add_executable(Kmeans kmeans/kmeans.cpp)
target_link_libraries(Kmeans husky)
target_link_libraries(Kmeans husky-ml)
if(USE_PROFILER)
target_link_libraries(Kmeans profiler)
endif(USE_PROFILER)
target_link_libraries(Kmeans ${HUSKY_EXTERNAL_LIB})
husky_default_properties(Kmeans)

# Kmeans_multi_stage
add_executable(Kmeans_multi_stage kmeans/kmeans_multi_stage.cpp)
target_link_libraries(Kmeans_multi_stage husky)
target_link_libraries(Kmeans_multi_stage husky-ml)
if(USE_PROFILER)
target_link_libraries(Kmeans_multi_stage profiler)
endif(USE_PROFILER)
target_link_libraries(Kmeans_multi_stage ${HUSKY_EXTERNAL_LIB})
husky_default_properties(Kmeans_multi_stage)

