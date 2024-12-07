import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Min max or mean #

tensor = torch.rand(2, 3, 4)
tensor_min = tensor.min()
tensor_max = tensor.max()
tensor_mean = tensor.mean()
tensor_sum = tensor.sum()

print(f"This is an original tensor:\n{tensor}")
print(f"\n Min in this tensor is: {tensor_min}")
print(f"\n Max in this tensor is: {tensor_max}")
print(f"\n Mean in this tensor is: {tensor_mean}")
print(f"\n Sum of this tensor is: {tensor_sum}")
