# 设置 CMake 最低版本要求为 3.18
cmake_minimum_required(VERSION 3.18)

# 定义工程名
project(CUDA_Course_Project LANGUAGES CUDA CXX)

# 使用 C++14
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Debug 模式下给 nvcc 添加调试参数
set(CMAKE_CUDA_FLAGS_DEBUG "-G -O0")

# 自动扫描所有课程目录下的 .cu 文件
file(GLOB_RECURSE COURSE_SOURCES
    course1/*.cu
)

foreach(src ${COURSE_SOURCES})
    get_filename_component(target ${src} NAME_WE)
    add_executable(${target} ${src})
endforeach()
