import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tensor = torch.rand(1, 3, 4)

print(f"Original tensor:\n{tensor}")
print(f"The shape of orriginal tensoris: {tensor.shape}")

reshaped_tensor = tensor.reshape(2, 6) # This two numbers multiplied needs to eaqule original number of elements in a tensor #
view_tensor = tensor.view(6, 2)
stack_tensor = torch.stack([tensor, tensor, tensor])
print(f"Reshaped tensor:\n{reshaped_tensor}")
print(f"The shape of switched tensoris: {reshaped_tensor.shape}")
print(f"Reviewed tensor:\n{view_tensor}")
print(f"The shape of reviewed tensoris: {view_tensor.shape}")
print(f"Stacke tensor:\n{stack_tensor}")
print(f"The shape of Stacked tensoris: {stack_tensor.shape}")
