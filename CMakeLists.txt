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
    course2/*.cu
    course3/*.cu
    course4/*.cu
    course5_1/*.cu
)

foreach(src ${COURSE_SOURCES})
    get_filename_component(target ${src} NAME_WE)
    get_filename_component(src_dir_name ${src} DIRECTORY)

    # 提取目录名（例如 course1, course2 等）
    get_filename_component(dir_name ${src_dir_name} NAME)

    add_executable(${target} ${src})

    # 设置输出目录为 build/course1, build/course2 等
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${dir_name}"
    )

    # 关键：链接 cuBLAS 库
    target_link_libraries(${target} PRIVATE cublas)
endforeach()
