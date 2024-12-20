import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
import time

# Load MNIST dataset using Hugging Face `datasets` library
mnist = load_dataset("mnist")

# Preprocessing: Compose transformations
transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))  # Normalize to mean=0.5, std=0.5
])

# Collate function for DataLoader (no need for Image.fromarray)
def collate_fn(batch):
    images = torch.stack([transform(x["image"]) for x in batch])  # Use ToTensor directly
    labels = torch.tensor([x["label"] for x in batch])
    return images, labels

# DataLoaders with lazy loading
BATCH_SIZE = 256
train_dataset = mnist["train"]
test_dataset = mnist["test"]

load_tr_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
load_test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Define the model
class Big_ass_module(nn.Module):
    def __init__(self, input_shape: int, hidden_layers: int, output_shape: int):
        super().__init__()
        self.stack_lay = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_layers),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layers, out_features=output_shape),
        )

    def forward(self, x):
        return self.stack_lay(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model_6 = Big_ass_module(input_shape=784, hidden_layers=128, output_shape=10).to(device)

# Loss and optimizer
LR = 0.01
loss_func = nn.CrossEntropyLoss()
opt_func = torch.optim.SGD(params=model_6.parameters(), lr=LR)

# Training loop
epochs = 10000
torch.manual_seed(47)

train_losses = []
test_losses = []
test_accuracies = []

train_time_start = time.time()

for epoch in tqdm(range(epochs)):
    train_loss = 0

    # Training phase
    for batch, (q, a) in enumerate(load_tr_data):
        q, a = q.to(device), a.to(device)
        model_6.train()
        y_pred = model_6(q)
        loss = loss_func(y_pred, a)
        train_loss += loss.item()

        opt_func.zero_grad()
        loss.backward()
        opt_func.step()

    train_loss /= len(load_tr_data)
    train_losses.append(train_loss)

    # Testing phase
    test_loss, test_acc = 0, 0
    model_6.eval()

    with torch.no_grad():
        for q, a in load_test_data:
            q, a = q.to(device), a.to(device)
            test_pred = model_6(q)
            test_loss += loss_func(test_pred, a).item()
            test_acc += (test_pred.argmax(dim=1) == a).sum().item()

    test_loss /= len(load_test_data)
    test_acc /= len(test_dataset)

    test_losses.append(test_loss)
    test_accuracies.append(test_acc * 100)

    # Print epoch statistics
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    if test_acc > 0.95:
        break

train_time_end = time.time()
print(f"Time that it took: {train_time_end - train_time_start:.2f} seconds")

# Plotting results
completed_epochs = len(train_losses)

plt.figure(figsize=(12, 4))

# Plot train and test loss
plt.subplot(1, 2, 1)
plt.plot(range(1, completed_epochs + 1), train_losses, label="Train Loss", marker="o")
plt.plot(range(1, completed_epochs + 1), test_losses, label="Test Loss", marker="o")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

# Plot test accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, completed_epochs + 1), test_accuracies, label="Test Accuracy", marker="o", color="green")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()

# Show plots
plt.tight_layout()
plt.show()
