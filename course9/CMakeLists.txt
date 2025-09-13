add_executable(fa flash_attn.cu)

add_executable(sa self_attn.cu)
target_link_libraries(fa PRIVATE ${CUDA_LIBRARIES} cublas)

target_link_libraries(sa PRIVATE ${CUDA_LIBRARIES} cublas)
