import torch
from pathlib import Path
import random
from torch import nn
import matplotlib.pyplot as plt

# Dataset parameters
weight = 50.98  # Slope of the line
bias = 2.4897   # Intercept

# Generating dataset
start = 1
end = 1000
step = 0.01
X = torch.arange(start, end, step, dtype=torch.float32).unsqueeze(dim=1)
y = weight * X + bias  # Correct equation for linear data generation

# Split data into training and testing sets
training_data = int(0.8 * len(X))
training_q, training_a = X[:training_data], y[:training_data]
testing_q, testing_a = X[training_data:], y[training_data:]

# Model definition
class Not_lin_reg(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize weights and biases with zeros for better convergence
        self.weights = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weights + self.bias  # Correct formula for linear regression

# Set seed for reproducibility
torch.manual_seed(13)

# Initialize model
model_1 = Not_lin_reg()

# Loss function and optimizer
loss_func nn.MSELoss()  # Mean Squared Error
optim_func = torch.optim.Adam(model_1.parameters(), lr=0.01)  # Lower learning rate

# Training loop
epochs = 1000000
losses = []  # To store loss values for plotting
losses_t = []
for epoch in range(epochs):
    model_1.train()  # Set model to training mode
    train_prediction = model_1(training_q)
    loss = loss_func(train_prediction, training_a)
    optim_func.zero_grad()
    loss.backward()
    optim_func.step()

    # Record the loss
    losses.append(loss.item())

    # Stop training if loss is very small
    if loss.item() < 1e-6:
        print("\033[92mStopping training as loss is sufficiently low.\033[0m")
        break

    # Evaluate the model periodically
    
    model_1.eval()  # Set model to evaluation mode
    with torch.no_grad():
        test_prediction = model_1(testing_q)
        test_loss = loss_func(test_prediction, testing_a)
        losses_t.append(test_loss.item())
        if epoch % 100 == 0:
            print(f"Epoch: {epoch+1} | Loss: {loss.item():.6f} | Test loss: {test_loss.item():.6f}")

# Print final model parameters
print("\nFinal Model Parameters:")
print(model_1.state_dict())


### Saving model ###

# Model dir #
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Model path #

MODEL_NAME = "01_Own_model.pth"
MODEL_SAVE_PATH = MODEL_DIR / MODEL_NAME

# Save the state_dict() of model #

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)

# Loading the same model (for sport) #

loaded_model_1 = Not_lin_reg()

loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print(f"Loaded model state_dict() is:\n{loaded_model_1.state_dict()}")

# Visualization function
def plot_pred(tr_data=training_q,
              tr_ans=training_a,
              te_data=testing_q,
              te_ans=testing_a,
              pred=None):
    plt.figure(figsize=(10, 10))
    plt.scatter(tr_data.numpy(), tr_ans.numpy(), c="r", s=10, label="Training Data")  # Training data in red
    plt.scatter(te_data.numpy(), te_ans.numpy(), c="b", s=10, label="Testing Data")  # Testing data in blue

    if pred is not None:
        plt.scatter(te_data.numpy(), pred.numpy(), c="g", s=10, label="Model Predictions")

    # Add the predicted line for training data
    plt.plot(tr_data.numpy(), (tr_data * model_1.weights + model_1.bias).detach().numpy(), 
             c="g", linewidth=2, label="Predicted Line")

    plt.title("Train-Test Split Visualization", fontsize=16)
    plt.xlabel("X values", fontsize=14)
    plt.ylabel("y values", fontsize=14)
    plt.legend(prop={"size": 14})
    plt.grid(True)
    plt.show()

# Loss curve plot
def plot_loss_curve(losses, losses_t):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses, label="Training Loss", color="blue")
    plt.title("Loss Curve", fontsize=16)
    plt.plot(range(len(losses_t)), losses_t, label="Testing Loss", color="red")
    plt.title("Loss Curve", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()

# Visualize predictions
model_1.eval()
with torch.no_grad():
    test_prediction = model_1(testing_q)

plot_pred(pred=test_prediction)
plot_loss_curve(losses, losses_t)

