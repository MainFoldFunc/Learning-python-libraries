import numpy as np
import torch
from torch import nn # Contains some graphs
import pandas as pd
import matplotlib.pyplot as plt

what_covering = {1 : "Data prepering",
                 2 : "Building model",
                 3 : "Training the model",
                 4 : "Saving and loading the model",
                 5 : "Evaluating",
                 6 : "Putting it all togheter"}

print(f"What I am doing in this chapter:\n")
for index, element in what_covering.items():
    print(f"{index} - {element}\n")
