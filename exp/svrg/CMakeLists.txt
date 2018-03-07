# LogisticRegression using configurable SVRG
add_executable(SVRG svrg.cpp)
target_link_libraries(SVRG husky)
target_link_libraries(SVRG husky-ml)
target_link_libraries(SVRG ${HUSKY_EXTERNAL_LIB})
husky_default_properties(SVRG)

# LogisticRegression using auto-parallelism SVRG
add_executable(AutoSVRG auto_svrg.cpp)
target_link_libraries(AutoSVRG husky)
target_link_libraries(AutoSVRG husky-ml)
target_link_libraries(AutoSVRG ${HUSKY_EXTERNAL_LIB})
husky_default_properties(AutoSVRG)

# LogisticRegression using chunk based configurable SVRG
add_executable(ChunkSVRG chunk_svrg.cpp)
target_link_libraries(ChunkSVRG husky)
target_link_libraries(ChunkSVRG husky-ml)
target_link_libraries(ChunkSVRG ${HUSKY_EXTERNAL_LIB})
husky_default_properties(ChunkSVRG)

# LogisticRegression using chunk based auto-parallelism SVRG
add_executable(AutoChunkSVRG auto_chunk_svrg.cpp)
target_link_libraries(AutoChunkSVRG husky)
target_link_libraries(AutoChunkSVRG husky-ml)
target_link_libraries(AutoChunkSVRG ${HUSKY_EXTERNAL_LIB})
husky_default_properties(AutoChunkSVRG)
