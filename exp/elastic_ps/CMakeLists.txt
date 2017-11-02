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
