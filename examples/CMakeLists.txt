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


include_directories(${PROJECT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})

if(LIBHDFS3_FOUND)
    include_directories(${LIBHDFS3_INCLUDE_DIR})
    add_definitions(-DWITH_HDFS)
endif(LIBHDFS3_FOUND)

if(MONGOCLIENT_FOUND)
    include_directories(${MONGOCLIENT_INCLUDE_DIR})
    add_definitions(-DWITH_MONGODB)
endif(MONGOCLIENT_FOUND)

# set(husky husky-lib husky-io husky-core husky-base) 
set(proj core base husky) 

### Applications

# ProcMain
add_executable(ProcMain ProcMain.cpp)
target_link_libraries(ProcMain ${proj})
target_link_libraries(ProcMain ${EXTERNAL_LIB})
set_property(TARGET ProcMain PROPERTY CXX_STANDARD 14)

# MasterMain
add_executable(MasterMain MasterMain.cpp)
target_link_libraries(MasterMain ${proj})
target_link_libraries(MasterMain ${EXTERNAL_LIB})
set_property(TARGET MasterMain PROPERTY CXX_STANDARD 14)

# ProcMainWithContext
add_executable(ProcMainWithContext ProcMainWithContext.cpp)
target_link_libraries(ProcMainWithContext ${proj})
target_link_libraries(ProcMainWithContext ${EXTERNAL_LIB})
set_property(TARGET ProcMainWithContext PROPERTY CXX_STANDARD 14)

# ProcMainWithContextDriver
add_executable(ProcMainWithContextDriver ProcMainWithContextDriver.cpp)
target_link_libraries(ProcMainWithContextDriver ${proj})
target_link_libraries(ProcMainWithContextDriver ${EXTERNAL_LIB})
set_property(TARGET ProcMainWithContextDriver PROPERTY CXX_STANDARD 14)

# MasterMainWithContext
add_executable(MasterMainWithContext MasterMainWithContext.cpp)
target_link_libraries(MasterMainWithContext ${proj})
target_link_libraries(MasterMainWithContext ${EXTERNAL_LIB})
set_property(TARGET MasterMainWithContext PROPERTY CXX_STANDARD 14)

# PI
add_executable(PI pi.cpp)
target_link_libraries(PI ${proj})
target_link_libraries(PI ${EXTERNAL_LIB})
set_property(TARGET PI PROPERTY CXX_STANDARD 14)
