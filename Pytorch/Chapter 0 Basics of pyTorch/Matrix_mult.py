import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Elemnt wise matrix multiplications #

#tensor = torch.tensor([1, 2, 3])
#tensor_2 = torch.tensor([4, 5, 6])
#matrix_mult_prod = torch.matmul(tensor, tensor_2) # This is the same #
#matrix_mult_prd = tensor @ tensor_2               # This are the same methodes #
#print(f"Normal way Tensor 1 * Tensor 2:\n{tensor * tensor_2}")
#print(f"Matrix way Tensor 1 * Tensor 2:\n{matrix_mult_prod}")

# Tensor re struct #

Tensor_A = torch.tensor([[1, 2, 3],
                        [4, 5, 6]])
Tensor_B = torch.tensor([[7, 8, 9],
                         [10, 11, 12]])
try:
    Tensor_res = torch.mm(Tensor_A, Tensor_B)
    print(Tensos_res)
except:
    print("You can't do that")


Tensor_B = Tensor_B.T # .T Swiches dimensions from (2, 3) to (2, 3)
Tensor_res = torch.mm(Tensor_A, Tensor_B)
print(Tensor_res)

