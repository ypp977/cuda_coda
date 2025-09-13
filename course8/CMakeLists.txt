add_executable(rmsnorm rmsnorm.cu)
target_link_libraries(rmsnorm PRIVATE ${CUDA_LIBRARIES})
