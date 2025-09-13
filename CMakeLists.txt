# 设置 CMake 最低版本要求为 3.18
# CUDA 在 CMake 3.17+ 才有较好的原生支持，这里要求 3.18 保证功能完整
cmake_minimum_required(VERSION 3.18)

# 定义工程名为 CUDA_Course_Project
# LANGUAGES 指定本项目会用到 CUDA 和 C++ 两种语言
project(CUDA_Course_Project LANGUAGES CUDA CXX)

# 指定 CUDA 语言标准为 C++14
# 默认 CUDA 代码可能只支持较旧标准，这里显式要求 C++14
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 自动扫描所有课程目录下的 .cu 文件
file(GLOB_RECURSE COURSE_SOURCES
    course1/*.cu
)

foreach(src ${COURSE_SOURCES})
    # 从文件名生成可执行文件名
    get_filename_component(target ${src} NAME_WE)
    add_executable(${target} ${src})
endforeach()
