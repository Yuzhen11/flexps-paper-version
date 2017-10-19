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

# MultiTasksExp
add_executable(MultiTasksExp multi_tasks_exp.cpp)
target_link_libraries(MultiTasksExp husky)
target_link_libraries(MultiTasksExp husky-ml)
target_link_libraries(MultiTasksExp ${HUSKY_EXTERNAL_LIB})
husky_default_properties(MultiTasksExp)

# MultiTasksLRExp
add_executable(MultiTasksLRExp multi_tasks_lr_exp.cpp)
target_link_libraries(MultiTasksLRExp husky)
target_link_libraries(MultiTasksLRExp husky-ml)
target_link_libraries(MultiTasksLRExp ${HUSKY_EXTERNAL_LIB})
husky_default_properties(MultiTasksLRExp)

# ParameterTuning
add_executable(ParameterTuning parameter_tuning.cpp)
target_link_libraries(ParameterTuning husky)
target_link_libraries(ParameterTuning husky-ml)
target_link_libraries(ParameterTuning ${HUSKY_EXTERNAL_LIB})
husky_default_properties(ParameterTuning)

# ScalabilityTest
add_executable(ScalabilityTest num_threads_tuning.cpp)
target_link_libraries(ScalabilityTest husky)
target_link_libraries(ScalabilityTest husky-ml)
target_link_libraries(ScalabilityTest ${HUSKY_EXTERNAL_LIB})
husky_default_properties(ScalabilityTest )

# ElasticPS
add_executable(ElasticPS elastic_ps.cpp)
target_link_libraries(ElasticPS husky)
target_link_libraries(ElasticPS husky-ml)
target_link_libraries(ElasticPS ${HUSKY_EXTERNAL_LIB})
husky_default_properties(ElasticPS)

# AutoElasticPS
add_executable(AutoElasticPS auto_elastic_ps.cpp)
target_link_libraries(AutoElasticPS husky)
target_link_libraries(AutoElasticPS husky-ml)
target_link_libraries(AutoElasticPS ${HUSKY_EXTERNAL_LIB})
husky_default_properties(AutoElasticPS)

# re-organize LR SVRG
add_executable(LR lrsvrg.cpp)
target_link_libraries(LR husky)
target_link_libraries(LR husky-ml)
target_link_libraries(LR ${HUSKY_EXTERNAL_LIB})
husky_default_properties(LR)
