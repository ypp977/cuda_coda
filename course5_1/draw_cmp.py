import numpy as np
import matplotlib.pyplot as plt

# 读取 CSV 文件（跳过表头）
data2 = np.genfromtxt('/home/test_fss/code/cuda_code/build/course5_1/sgemm_benchmark_v3.csv', delimiter=',', skip_header=1)
data1 = np.genfromtxt('/home/test_fss/code/cuda_code/build/course5_1/sgemm_benchmark_v4.csv', delimiter=',', skip_header=1)

# 提取三列数据
sizes = data1[:, 0]          # Size
mygemm1_time = data1[:, 2]    # MySGEMM_v1_Time_ms
mygemm2_time = data2[:, 2]    # MySGEMM_v1_Time_ms
print(mygemm1_time)
print(mygemm2_time)
ratio =  mygemm2_time / mygemm1_time /1.
print(ratio)
# 创建图表
plt.figure(figsize=(10, 6))

# 绘图
plt.plot(sizes, mygemm1_time, label="mygemm4 GFLOPS", marker='o', color='blue')
plt.plot(sizes, mygemm2_time, label="mygemm3 GFLOPS", marker='s', color='orange')

# 设置图表标题和坐标轴标签
plt.title("MySGEMM_v3 vs MySGEMM_v4 GFLOPS", fontsize=14)
plt.xlabel("Matrix Size (N x N)", fontsize=12)
plt.ylabel("GFLOPS", fontsize=12)

# 设置 y 轴为对数刻度
# plt.yscale('log')

# 图例
plt.legend()

# 网格
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 自动调整布局
plt.tight_layout()

# 保存图像到文件（可选）
plt.savefig("cublas_vs_mysgemm_v4_comp.png", dpi=300)

# 显示图表
plt.show()
