import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tensor operations #

#tensor = torch.rand(2, 3, 4)
#tensor_2 = torch.rand(2, 3, 4)
#print(f"Tensor 1:\n{tensor}")
#print(f"Tensor 2:\n{tensor_2}")

#tensor_res_sum = tensor + tensor_2
#tensor_res_sub = tensor - tensor_2
#tensor_res_mul = tensor * tensor_2
#tensor_res_div = tensor / tensor_2

#print(f"tensor 1 + tensor 2 =\n{tensor_res_sum}")
#print(f"tensor 1 - tensor 2 =\n{tensor_res_sub}")
#print(f"tensor 1 * tensor 2 =\n{tensor_res_mul}")
#print(f"tensor 1 / tensor 2 =\n{tensor_res_div}")

tensor = torch.tensor([1, 2, 3])
print(f"Original tensor:\n{tensor}")

print(f"Tensor + 10:\n{tensor + 10}")
print(f"Tensor * 10:\n{tensor * 10}")
print(f"Tensor / 10:\n{tensor / 10}")
print(f"Tensor - 10:\n{tensor - 10}")

