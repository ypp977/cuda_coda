import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import extension_cpp
from extension_cpp.ops import myfla
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
import torch.nn as nn

import math
def get_tensors(BS, HEAD, SEQLEN, DIM, dtype=torch.float16):
    q = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    return q, k, v

def self_attention(q, k, v):
    q = q / math.sqrt(q.shape[-1])
    score = torch.matmul(q, k.transpose(-2, -1))
    s = torch.softmax(score, dim=-1)
    return s @ v


def main():
    BS, HEAD, SEQLEN, DIM = 1000, 1, 64, 128

    q, k, v = get_tensors(BS, HEAD, SEQLEN, DIM, dtype=torch.float32)

    # ----------------------------
    # 计时 self_attention
    # ----------------------------
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()  # 确保之前的计算都已完成
    start_event.record()
    o1 = self_attention(q, k, v)
    end_event.record()
    torch.cuda.synchronize()  # 等待 kernel 完成
    time_self_attn = start_event.elapsed_time(end_event)  # 单位是毫秒

    print("self_attention output:")
    print(o1)

    # ----------------------------
    # 计时 myfla
    # ----------------------------
    start_event.record()
    o2 = myfla(q, k, v)
    end_event.record()
    torch.cuda.synchronize()
    time_myfla = start_event.elapsed_time(end_event)

    print("myfla output:")
    print(o2)

    # ----------------------------
    # 验证结果是否一致
    # ----------------------------
    try:
        torch.testing.assert_close(o1, o2, rtol=1e-3, atol=1e-3)
        print("✅ 输出一致")
    except AssertionError as e:
        print("❌ 输出不一致")
        print(e)

    # ----------------------------
    # 打印耗时
    # ----------------------------
    print(f"self_attention 耗时: {time_self_attn:.4f} ms")
    print(f"myfla 耗时: {time_myfla:.4f} ms")


main()