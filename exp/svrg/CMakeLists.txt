# LogisticRegression using SVRG
add_executable(SVRG lrsvrg.cpp)
target_link_libraries(SVRG husky)
target_link_libraries(SVRG husky-ml)
target_link_libraries(SVRG ${HUSKY_EXTERNAL_LIB})
husky_default_properties(SVRG)
