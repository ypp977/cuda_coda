#!/bin/bash

# ================================
# 用量定义区域 (Configuration)
# ================================

# CUDA 安装路径（可根据需要切换版本）
CUDA_HOME="/home/test_fss/cuda-home/12.8"

# NCU 工具路径
NCU_BIN="${CUDA_HOME}/bin/ncu"

# 被测程序路径
APP_BIN="/home/test_fss/code/cuda_code/build/course5_1/matmul3"

# 输出文件名（不含扩展名，ncu 会自动加 .ncu-rep）
OUTPUT_NAME="matmul3"

# NCU 配置参数
NCU_CONFIG=(
    --config-file off
    --export "${OUTPUT_NAME}"
    --force-overwrite
    --set full
)

# ================================
# 执行命令
# ================================

echo "Starting NCU profiling..."
echo "Target: ${APP_BIN}"
echo "Output: ${OUTPUT_NAME}.ncu-rep"
echo "Command: sudo ${NCU_BIN} ${NCU_CONFIG[*]} ${APP_BIN}"

# 执行分析
sudo "${NCU_BIN}" "${NCU_CONFIG[@]}" "${APP_BIN}"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "Profiling completed successfully."
else
    echo "Profiling failed!"
    exit 1
fi
