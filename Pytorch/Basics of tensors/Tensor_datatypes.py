import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

float_tensor_32 = torch.tensor([3.0, 6.0, 9.0],
                            dtype=None, # It is a datatype of a tensor (float32, float16, float64) #
                            device=None,   # On wich device will this tensor by on (cpu, gpu) 
                            requires_grad=False) #

float_tensor_16 = float_tensor_32.type(torch.half)
print(f"This is 16 byte tensor:\n{float_tensor_16}")


