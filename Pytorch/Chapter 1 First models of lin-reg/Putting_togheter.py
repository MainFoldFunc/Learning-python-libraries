import torch
from pathlib import Path
import numpy
from torch import nn
import matplotlib.pyplot as plt

# Define training loop
def training_loop(tr_q, tr_a, opt_func, loss_func, model_2):
    model_2.train()
    train_pred = model_2(tr_q)
    loss = loss_func(train_pred, tr_a)
    loss.backward()
    opt_func.step()
    opt_func.zero_grad()
    return loss.item()

# Define testing loop
def testing_loop(ts_q, ts_a, epoch, losses_t, loss_func, model_2):
    with torch.no_grad():
        test_pred = model_2(ts_q)
        test_loss = loss_func(test_pred, ts_a)
        losses_t.append(test_loss.item())
        if epoch % 100 == 0:
            print(f"\033[92mEpoch: {epoch} | Test loss: {test_loss.item():.10f}")

# Data preparation
weight = 3.98
bias = 1.01
start, end, step = 0, 10, 0.03
X = torch.arange(start, end, step, dtype=torch.float32).unsqueeze(dim=1)
y = weight * X + bias
training_prt = int(len(X) * 0.8)
training_q, training_a = X[:training_prt], y[:training_prt]
testing_q, testing_a = X[training_prt:], y[training_prt:]

# Model definition
class Lin_reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Training setup
torch.manual_seed(42)
model_2 = Lin_reg()
loss_func = nn.MSELoss()
opt_func = torch.optim.Adam(model_2.parameters(), lr=0.01)
losses, losses_t = [], []

# Training loop
loss = 1
epoch = 0
while loss > 1e-10:
    epoch += 1
    loss = training_loop(training_q, training_a, opt_func, loss_func, model_2)
    losses.append(loss)
    model_2.eval()
    testing_loop(testing_q, testing_a, epoch, losses_t, loss_func, model_2)

print(f"Final Model Params:")
print(model_2.state_dict())

# Model dir #
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Model path #

MODEL_NAME = "02_Own_model.pth"
MODEL_SAVE_PATH = MODEL_DIR / MODEL_NAME

# Save the state_dict() of model #

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(), f=MODEL_SAVE_PATH)



# Plot predictions
def plot_pred(tr_data=training_q, tr_ans=training_a, te_data=testing_q, te_ans=testing_a, pred=None):
    plt.figure(figsize=(10, 10))
    plt.scatter(tr_data.numpy(), tr_ans.numpy(), c="r", s=10, label="Training Data")
    plt.scatter(te_data.numpy(), te_ans.numpy(), c="b", s=10, label="Testing Data")
    if pred is not None:
        plt.scatter(te_data.numpy(), pred.numpy(), c="g", s=10, label="Model Predictions")
    plt.plot(tr_data.numpy(), 
             (tr_data * model_2.linear_layer.weight.item() + model_2.linear_layer.bias.item()), 
             c="g", linewidth=2, label="Predicted Line")
    plt.title("Train-Test Split Visualization", fontsize=16)
    plt.xlabel("X values", fontsize=14)
    plt.ylabel("y values", fontsize=14)
    plt.legend(prop={"size": 14})
    plt.grid(True)
    plt.show()

def plot_loss_curve(losses, losses_t):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses, label="Training Loss", color="blue")
    plt.plot(range(len(losses_t)), losses_t, label="Testing Loss", color="red")
    plt.title("Loss Curve", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()

# Visualize predictions
model_2.eval()
with torch.no_grad():
    test_pred = model_2(testing_q)
plot_pred(pred=test_pred)
plot_loss_curve(losses, losses_t)

