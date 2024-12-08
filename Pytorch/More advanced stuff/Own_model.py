import torch
from torch import nn
import matplotlib.pyplot as plt

# Dataset parameters
weight = 5
bias = 2
start = 1
end = 1000000
X = torch.arange(start, end, dtype=torch.float32).unsqueeze(dim=1)
y = X**2 + weight * bias

# Split data into training and testing sets
training_data = int(0.8 * len(X))
training_q, training_a = X[:training_data], y[:training_data]
testing_q, testing_a = X[training_data:], y[training_data:]

# Model definition
class Not_lin_reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x**2 + self.weights * self.bias

# Set seed for reproducibility
torch.manual_seed(13)

# Initialize model
model_1 = Not_lin_reg()

# Loss and optimizer
loss_func = nn.L1Loss()
optim_func = torch.optim.Adam(model_1.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model_1.train()  # Set model to training mode
    train_prediction = model_1(training_q)
    loss = loss_func(train_prediction, training_a)
    optim_func.zero_grad()
    loss.backward()
    optim_func.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate the model
model_1.eval()  # Set model to evaluation mode
with torch.inference_mode():
    answer_prediction = model_1(testing_q)

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
        # Ensure shapes align before plotting
        if te_data.shape[0] == pred.shape[0]:
            plt.scatter(te_data.numpy(), pred.numpy(), c="g", s=10, label="Model Predictions")
        else:
            print("Shape mismatch: te_data and pred must have the same number of elements.")

    plt.title("Train-Test Split Visualization", fontsize=16)
    plt.xlabel("X values", fontsize=14)
    plt.ylabel("y values", fontsize=14)
    plt.legend(prop={"size": 14})
    plt.grid(True)
    plt.show()

# Visualize predictions
plot_pred(pred=answer_prediction)

