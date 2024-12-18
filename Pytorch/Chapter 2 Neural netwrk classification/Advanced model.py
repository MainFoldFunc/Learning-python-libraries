import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Creating dataset
Classes = 9
Num_Of_Features = 2
Random_seed = 13

q, a = make_blobs(n_samples=2000,
                  n_features=Num_Of_Features,
                  centers=Classes,
                  cluster_std=1,
                  random_state=Random_seed)

train_q, test_q, train_a, test_a = train_test_split(q, a,
                                                    test_size=0.2,
                                                    random_state=Random_seed)

# Plotting the toy data
plt.figure(figsize=(10, 7))
plt.scatter(q[:, 0], q[:, 1], c=a, cmap=plt.cm.RdYlBu)
plt.title("Generated Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
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

model_4 = multi_model(input_features=2, output_features=9, hidden_layers=64)
model_4 = model_4.to(device)

# Define loss function and optimizer
loss_func = nn.CrossEntropyLoss()
opt_func = torch.optim.Adam(model_4.parameters(), lr=0.001)

## Accuracy function ##
def acc(y_true, y_pred):
    corr = torch.eq(y_true, y_pred).sum().item()
    return (corr / len(y_pred)) * 100

### Training loop ###
epochs = 50000

loss_values_for_plot = []
test_values_for_plot = []

for epoch in range(epochs):
    model_4.train()

    logits = model_4(train_q)
    loss = loss_func(logits, train_a)
    loss_values_for_plot.append(loss.item())

    opt_func.zero_grad()
    loss.backward()
    opt_func.step()

    model_4.eval()
    with torch.no_grad():
        # Training accuracy
        train_logits = model_4(train_q)
        train_pred = torch.argmax(train_logits, dim=1)
        train_acc = acc(train_a, train_pred)

        # Testing accuracy
        test_logits = model_4(test_q)
        test_pred = torch.argmax(test_logits, dim=1)
        test_acc = acc(test_a, test_pred)
        test_values_for_plot.append(test_acc)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch + 1} / {epochs} \t Train acc: {train_acc:.4f}% \t Test acc: {test_acc:.4f}%")

    if test_acc > 99: 
        print("Early stopping because testing accuracy exceeded 97%")
        break

# Plotting training loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(loss_values_for_plot, label="Training Loss", color='blue')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# Plotting test accuracy over epochs
plt.figure(figsize=(10, 6))
plt.plot(test_values_for_plot, label="Test Accuracy", color='green')
plt.title("Test Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()
plt.show()

# Comparing predictions with actual data
with torch.no_grad():
    test_preds = model_4(test_q).argmax(dim=1)

plt.figure(figsize=(10, 7))
plt.scatter(test_q[:, 0].cpu(), test_q[:, 1].cpu(), c=test_preds.cpu(), cmap=plt.cm.RdYlBu, marker="o", label="Predicted")
plt.scatter(test_q[:, 0].cpu(), test_q[:, 1].cpu(), c=test_a.cpu(), cmap=plt.cm.RdYlBu, marker="x", label="Actual", alpha=0.5)
plt.title("Predictions vs Actual")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

