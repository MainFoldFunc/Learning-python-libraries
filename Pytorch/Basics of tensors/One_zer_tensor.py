import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tensor of all zeroes and ones #

zero_tensor = torch.zeros(size=(1, 3, 4))
print(f"The only zero tensor:\n{zero_tensor}")

one_tensor = torch.ones(size=(1, 3, 4))
print(f"The only zero tensor:\n{one_tensor}")

print(f"What is a data type of this tensor: {one_tensor.dtype}")
print(f"What is a data type of this tensor: {zero_tensor.dtype}")
