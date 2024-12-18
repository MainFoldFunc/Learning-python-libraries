import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor  # Corrected import
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

## Getting data ##
train_d = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
test_d = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

### Loading data into the model ###
BATCH_SIZE = 32

train_dataLoad = DataLoader(  # Corrected to 'DataLoader'
    dataset=train_d,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_dataLoad = DataLoader(  # Corrected to 'DataLoader' and 'shuffle=False'
    dataset=test_d,
    batch_size=BATCH_SIZE,
    shuffle=False
)

### Create model ###
class MNIST_module(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_layers: int,  # Fixed spelling typo
                 output_shape: int):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_layers),  # Fixed 'hidden_units'
            nn.ReLU(),  # Adding ReLU activation
            nn.Linear(in_features=hidden_layers, out_features=output_shape)  # Fixed 'hidden_units'
        )

    def forward(self, x):
        return self.layer_stack(x)

model_5 = MNIST_module(input_shape=784,  # 28x28 images flattened
                       hidden_layers=64,  # Number of hidden units
                       output_shape=10).to("cpu")  # 10 classes in FashionMNIST

LR = 0.1

opt_func = torch.optim.SGD(params=model_5.parameters(), lr = LR)
