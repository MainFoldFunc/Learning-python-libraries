import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Random tensors #

rand_tensor = torch.rand(2, 3, 4)

print(f"The random tensor looks like this: \n{rand_tensor}")
print(f"The random tensor dimensions: \n{rand_tensor.ndim}")

# Create a tensor similar to an imgae #

i_tensor = torch.rand(size=(256, 256, 3)) # height, width, base colours #

print(f"{i_tensor.shape}\n{i_tensor.ndim}")

# Tensor of all zeroes and ones #

zero_one_tensor = torch.zeros(sze=(1, 3, 4))
print(f"The only zero tensor:\n{zero_one_tensor}")
