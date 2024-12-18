import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Creating dataset
Classes = 4
Num_Of_Features = 2
Random_seed = 47

q, a = make_blobs(n_samples=2000,
                  n_features=Num_Of_Features,
                  centers=Classes,
                  cluster_std=2.3,
                  random_state=Random_seed)

train_q, test_q, train_a, test_a = train_test_split(q, a,
                                                    test_size=0.2,
                                                    random_state=Random_seed)

# Plotting the toy data
plt.figure(figsize=(10, 7))
plt.scatter(q[:, 0], q[:, 1], c=a, cmap=plt.cm.RdYlBu)
plt.show()

# Converting numpy arrays to torch tensors
train_q = torch.from_numpy(train_q).type(torch.float32)
train_a = torch.from_numpy(train_a).type(torch.long)
test_q = torch.from_numpy(test_q).type(torch.float32)
test_a = torch.from_numpy(test_a).type(torch.long)

# Device initialization
device = "cuda" if torch.cuda.is_available() else "cpu"

# Building the model
class multi_model(nn.Module):
    def __init__(self, input_features, output_features, hidden_layers=64):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_layers),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layers, out_features=hidden_layers),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layers, out_features=hidden_layers),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layers, out_features=output_features)
        )
    def forward(self, x):
        return self.linear_layer_stack(x)

model_4 = multi_model(input_features=2, output_features=4, hidden_layers=64)
model_4 = model_4.to(device)

