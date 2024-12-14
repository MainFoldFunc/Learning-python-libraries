import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Torch range #

torch_range = torch.arange(start = 0, end = 10, step=2)
print(f"This is a tensor range: {torch_range}")

# Tensor like #

like_tensors = torch.zeros_like(input=torch_range)
print(f"This is a lkie tensor: \n{like_tensors}")
