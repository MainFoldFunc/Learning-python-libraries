import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Numpy in Torch #

tensor = torch.rand(1, 3, 4)
np_array = np.array([[1, 2, 3, 4],
                     [1, 2, 3, 4]])
print(f"Original tensor:\n{tensor}")
print(f"Original array: \n{np.array}")

arr_to_ten = torch.from_numpy(np_array) # Converts to float64 not float32. #
print(f"The tensor to the array is: \n{arr_to_ten}")
