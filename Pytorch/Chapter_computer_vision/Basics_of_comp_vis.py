import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor  # Corrected import
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import requests
from pathlib import Path
from helper_functions import accuracy_fn 
from helper_functions import timing
from tqdm.auto import tqdm
import time  # Add this import for timing

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

if Path("helper_functions.py").is_file():
    print(f"requirements already satisfied")
else:
    print(f"Downloading...")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

loss_func = nn.CrossEntropyLoss()  # Fixed 'CrosEntropyLoss' to 'CrossEntropyLoss'
opt_func = torch.optim.SGD(params=model_5.parameters(), lr=LR)  # Fixed model parameters

epochs = 50000

torch.manual_seed(47)

# Timing the training
train_time_start = time.time()

for epoch in tqdm(range(epochs)):
    train_loss = 0

    for batch, (q, a) in enumerate(train_dataLoad):  # Fixed to train_dataLoad
        model_5.train()
        y_pred = model_5(q)
        loss = loss_func(y_pred, a)
        train_loss += loss.item()  # Use .item() to get the scalar value from the loss tensor

        opt_func.zero_grad()
        loss.backward()
        opt_func.step()

    if batch % 400 == 0:
        print(f"Looked at {batch * len(q)} / {len(train_dataLoad.dataset)} samples")  # Corrected 'Lokked' to 'Looked'

    train_loss /= len(train_dataLoad)

    # Testing the model
    test_loss, test_acc = 0, 0
    model_5.eval()

    with torch.no_grad():  # Fixed 'torch.inference_mode()' to 'torch.no_grad()'
        for q, a in test_dataLoad:  # Corrected to test_dataLoad
            test_pred = model_5(q)
            test_loss += loss_func(test_pred, a).item()  # Use .item() to get scalar loss
            test_acc += accuracy_fn(y_true=a, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(test_dataLoad)
        test_acc /= len(test_dataLoad)

    print(f"\n Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Test_acc: {test_acc:.2f}")

    if test_loss > 97:
        print(f"stoping beacous test loss more than 97%")
        break

# Timing the end of training
train_time_stop = time.time()
timing(train_time_start, train_time_stop)
# Save the model
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_DIR / "MNIST_model.pth"

torch.save(model_5.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}") 
