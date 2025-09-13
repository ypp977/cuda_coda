import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import extension_cpp
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
import torch.nn as nn

size = 32
a1 = torch.randn(size, device='cuda')
a2 = torch.randn(size, device='cuda')
out = extension_cpp.ops.mymuladd(a1,a2,3.0)
print(out)