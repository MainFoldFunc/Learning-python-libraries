import torch
from torch import nn
import matplotlib.pyplot as plt

# Create known parameters
weight = 0.60
bias = 0.20

# Create data
start = 0
end = 1
step = 0.00001
X = torch.arange(start, end, step).unsqueeze(dim=1)  # Creating input data (column vector)
y = weight + X ** 2 + bias  # Generating corresponding output data

# Splitting data between the testing and training sets
tr_data = int(0.9 * len(X))  # 80% of the data for training
X_tr_q, y_tr_a = X[:tr_data], y[:tr_data]  # Training data
X_te_q, y_te_a = X[tr_data:], y[tr_data:]  # Testing data

# Simple First Linear Regression Model
class Lin_reg(nn.Module):  # nn.Module is essential to define custom PyTorch models
    def __init__(self):
        super().__init__()
        # Initialize weights and biases as trainable parameters
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass (compute predictions using the linear equation y = wx + b)
        return self.weights * x ** 2 + self.bias

# Create a random seed
torch.manual_seed(42)  # Ensure reproducibility of random initialization

model_0 = Lin_reg()  # Instantiate the model

# Making predictions (initial predictions for testing data)
with torch.inference_mode():
    y_pred = model_0(X_te_q)

### SETUP A LOSS FUNCTION ###
loss_fn = nn.L1Loss()

### SETUP OPTIMIZER ###
opt_fn = torch.optim.SGD(params=model_0.parameters(),
                         lr=0.01)

### LOOP ###
epochs = 1000  # Loop through the data

for epoch in range(epochs):
    model_0.train()  # Turns on gradient descent

    # Forward pass for training data
    train_pred = model_0(X_tr_q)  # Compute predictions for training data

    # Calculate loss
    loss = loss_fn(train_pred, y_tr_a)

    # Zero gradients before backpropagation
    opt_fn.zero_grad()

    # Backpropagation
    loss.backward()

    # Step optimizer
    opt_fn.step()

    # Model in evaluation mode
    model_0.eval()
    print(list(model_0.parameters()))

# Recalculate predictions for the testing data after training
with torch.inference_mode():
    y_pred = model_0(X_te_q)
print(weight)
print(bias)
print(list(model_0.parameters()))
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
        # Debug shapes
        print("te_data shape:", te_data.shape)
        print("pred shape:", pred.shape)

        # Ensure shapes align before plotting
        if te_data.shape[0] == pred.shape[0]:
            plt.scatter(te_data.numpy(), pred.detach().numpy(), c="g", s=10, label="Model Predictions")
        else:
            print("Shape mismatch: te_data and pred must have the same number of elements.")

    plt.title("Train-Test Split Visualization", fontsize=16)
    plt.xlabel("X values", fontsize=14)
    plt.ylabel("y values", fontsize=14)
    plt.legend(prop={"size": 14})
    plt.grid(True)
    plt.show()

# Call the function and visualize
plot_pred()

