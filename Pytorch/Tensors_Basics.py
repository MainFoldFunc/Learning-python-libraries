import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# print(torch.__version__)

# Scalars #

scalar = torch.tensor(7)
print(f"this tensor is {scalar}")
print(f"This tensor has {scalar.ndim} dimensions")

# Vector #

vector = torch.tensor([7, 7])
print(f"This is vector {vector}")
print(f"This tensor has {vector.ndim} dimensions")
print(f"This tensor has {vector.shape} shape")

# Matrix #

matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"This is a matrix:\n{matrix}")
print(f"This tensor has {matrix.ndim} dimensions")
print(f"This tensor has {matrix.shape} shape")
print(f"The first row of the matrix is:\n{matrix[0]}")

# TENSORS #

tensor = torch.tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]],
                       [[1, 2, 3],
                        [4, 5, 6],
                      [7, 8, 9]]])
print(f"This is a matrix:\n{tensor}")
print(f"This tensor has {tensor.ndim} dimensions")
print(f"This tensor has {tensor.shape} shape")
print(f"The first row of the matrix is:\n{tensor[0]}")


