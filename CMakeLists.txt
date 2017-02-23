# Copyright 2017 Husky Team
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

cmake_minimum_required(VERSION 3.0.2)

set(dep_DIR ${PROJECT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})


# use submodule
add_subdirectory(husky)

# use the definition
add_definitions(${HUSKY_EXTERNAL_DEFINITION})

# all the components
add_subdirectory(worker)
add_subdirectory(core)
add_subdirectory(kvstore)


# build libhusky-ml.a
set(husky-ml-objs $<TARGET_OBJECTS:mlworker-objs> $<TARGET_OBJECTS:mlcore-objs> 
                  $<TARGET_OBJECTS:mlkvstore-objs>)
add_library(husky-ml STATIC ${husky-ml-objs})

add_library(husky-ml-master STATIC $<TARGET_OBJECTS:mlcore-objs>)

# build the examples and test
add_subdirectory(examples)
add_subdirectory(test)
