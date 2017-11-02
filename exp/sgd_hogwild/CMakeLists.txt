add_executable(SGDHogwild sgd_hogwild.cpp)
target_link_libraries(SGDHogwild husky)
target_link_libraries(SGDHogwild husky-ml)
target_link_libraries(SGDHogwild ${HUSKY_EXTERNAL_LIB})
husky_default_properties(SGDHogwild)

