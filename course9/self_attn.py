import torch
import math
import numpy as np

def self_attention_pytorch(Q, K, V):
    """
    PyTorch equivalent of the CUDA self_attention function.
    """
    m, n = Q.shape
    sm_scale = 1.0 / math.sqrt(n)
    attn = torch.mm(Q, K.transpose(-2, -1)) * sm_scale  # [m, m]
    attn = attn.softmax(dim=-1)                        # row-wise softmax
    O = torch.mm(attn, V)                              # [m, n]
    return O

# -------------------------------
# 参数设置
# -------------------------------
m, n = 64, 128  # 可根据你的测试需求修改
dtype = torch.float32

# 设置随机种子以确保可复现性（便于和 CUDA 对比）
torch.manual_seed(42)

# 生成 Q, K, V (m, n)
Q = torch.randn(m, n, dtype=dtype)
K = torch.randn(m, n, dtype=dtype)
V = torch.randn(m, n, dtype=dtype)

# 可选：移动到 GPU 计算（结果仍可保存为 CPU numpy）
# Q, K, V = Q.cuda(), K.cuda(), V.cuda()

# -------------------------------
# 执行自注意力
# -------------------------------
with torch.no_grad():
    O = self_attention_pytorch(Q, K, V)


# -------------------------------
# 保存 Q, K, V, O 到 .bin 文件
# 使用 numpy 保存为 float32 格式
# -------------------------------

def save_tensor_bin(tensor, filename):
    """Save a PyTorch tensor as binary file (float32)."""
    tensor_np = tensor.cpu().numpy()  # 确保在 CPU 上并转为 numpy
    tensor_np.astype(np.float32).tofile(filename)
    print(f"Saved {filename} with shape {tensor.shape}, dtype {tensor_np.dtype}")

# 保存所有张量
save_tensor_bin(Q, "/home/test_fss/code/cuda_code/course9/Q.bin")
print(Q)
save_tensor_bin(K, "/home/test_fss/code/cuda_code/course9/K.bin")
save_tensor_bin(V, "/home/test_fss/code/cuda_code/course9/V.bin")
save_tensor_bin(O, "/home/test_fss/code/cuda_code/course9/O.bin")

print("✅ All tensors saved as .bin files.")
