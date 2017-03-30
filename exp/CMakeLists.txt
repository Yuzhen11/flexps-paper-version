include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/husky ${HUSKY_EXTERNAL_INCLUDE})

# ReadLocalGlobal
add_executable(ReadLocalGlobal read_local_global.cpp)
target_link_libraries(ReadLocalGlobal husky)
target_link_libraries(ReadLocalGlobal husky-ml)
target_link_libraries(ReadLocalGlobal ${HUSKY_EXTERNAL_LIB})
husky_default_properties(ReadLocalGlobal)

# LogisticRegression SVRG
add_executable(LRSVRG logistic_regression_svrg.cpp)
target_link_libraries(LRSVRG husky)
target_link_libraries(LRSVRG husky-ml)
target_link_libraries(LRSVRG ${HUSKY_EXTERNAL_LIB})
husky_default_properties(LRSVRG)

# ReadTrain
add_executable(ReadTrain read_train.cpp)
target_link_libraries(ReadTrain husky)
target_link_libraries(ReadTrain husky-ml)
target_link_libraries(ReadTrain ${HUSKY_EXTERNAL_LIB})
husky_default_properties(ReadTrain)
