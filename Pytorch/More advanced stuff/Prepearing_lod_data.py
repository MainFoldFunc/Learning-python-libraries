import torch
from torch import nn
import matplotlib.pyplot as plt

# Create known parameters
weight = 0.6
bias = 0.2

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)  # Creating input data (column vector)
y = weight * X + bias  # Generating corresponding output data

# Splitting data between the testing and training sets
tr_data = int(0.8 * len(X))  # 80% of the data for training
X_tr_q, y_tr_a = X[:tr_data], y[:tr_data]  # Training data
X_te_q, y_te_a = X[tr_data:], y[tr_data:]  # Testing data

# Uncomment to debug and print datasets
# print(f"This is X training dataset:\n{X_tr_q}")
# print(f"This is X testing dataset:\n{X_te_q}")
# print(f"This is y training dataset:\n{y_tr_a}")
# print(f"This is y testing dataset:\n{y_te_a}")

# Simple First Linear Regression Model
class Lin_reg(nn.Module):  # nn.Module is essential to define custom PyTorch models
    def __init__(self):
        super().__init__()
        # Initialize weights and biases as trainable parameters
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass (compute predictions using the linear equation y = wx + b)
        return self.weights * x + self.bias

# Checking the model
# Create a random seed
torch.manual_seed(42)  # Ensure reproducibility of random initialization

model_0 = Lin_reg()  # Instantiate the model
#print(list(model_0.parameters()))  # Print the model parameters to verify initialization

# Making predictions #

with torch.inference_mode():
    y_pred = model_0(X_te_q)


#print(f"Perfect data:\n{y_te_a}")
#print(f"Model data:\n{y_pred}")

# Optional: Uncomment the function below to visualize the data

# Plot the data
def plot_pred(tr_data=X_tr_q,
              tr_ans=y_tr_a,
              te_data=X_te_q,
              te_ans=y_te_a,
              pred=y_pred):
    plt.figure(figsize=(10, 10))
    plt.scatter(tr_data.numpy(), tr_ans.numpy(), c="r", s=10, label="Training Data")  # Training data in red
    plt.scatter(te_data.numpy(), te_ans.numpy(), c="b", s=10, label="Testing Data")  # Testing data in blue

    if pred is not None:
        # Plot predictions if provided (e.g., after training the model)
        plt.scatter(te_data.numpy(), pred.numpy(), c="g", s=10, label="Model Predictions")
    plt.title("Train-Test Split Visualization", fontsize=16)
    plt.xlabel("X values", fontsize=14)
    plt.ylabel("y values", fontsize=14)
    plt.legend(prop={"size": 14})
    plt.grid(True)
    plt.show()

# Uncomment to call the function and visualize
plot_pred()

