add_executable(naive_transpose naive_transpose.cu)
add_executable(transpose_bench   transpose_bench.cu)
target_link_libraries(naive_transpose PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(transpose_bench PRIVATE ${CUDA_LIBRARIES})
