file(GLOB lda-src-files
    lda_doc.cpp
    doc_sampler.cpp
    lda_stat.cpp)
husky_cache_variable(lda-src-files ${lda-src-files})
add_library(lda-objs STATIC ${lda-src-files})
husky_default_properties(lda-objs)

# LDA 
add_executable(LDAMain lda_main.cpp)
target_link_libraries(LDAMain husky)
target_link_libraries(LDAMain husky-ml)
target_link_libraries(LDAMain lda-objs)
target_link_libraries(LDAMain profiler)
target_link_libraries(LDAMain ${HUSKY_EXTERNAL_LIB})
husky_default_properties(LDAMain)
