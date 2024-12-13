import torch
from torch import nn
import matplotlib.pyplot as plt

# Dataset parameters
weight = 5
bias = 2

start = 1
end = 100
step = 0.01
X = torch.arange(start, end, step,  dtype=torch.float32).unsqueeze(dim=1)
y = weight + X * bias * X**2

# Normalize X and y
X_min, X_max = X.min(), X.max()
y_min, y_max = y.min(), y.max()
X = (X - X_min) / (X_max - X_min)
y = (y - y_min) / (y_max - y_min)

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
        return self.weights + x * self.bias * x ** 2

# Set seed for reproducibility
torch.manual_seed(13)

# Initialize model
model_1 = Not_lin_reg()

# Loss and optimizer
loss_func = nn.MSELoss()  # Using Mean Squared Error for better large-value handling
optim_func = torch.optim.Adam(model_1.parameters(), lr=0.1)  # Lower learning rate

# Training loop
epochs = 10000
for epoch in range(epochs):
    model_1.train()  # Set model to training mode
    train_prediction = model_1(training_q)
    loss = loss_func(train_prediction, training_a)
    optim_func.zero_grad()
    loss.backward()
    optim_func.step()

    # Stop training if loss reaches 0
    if loss.item() < 1e-100 and answer_loss.item() < 1e-100:
        print("\033[92mStopping training as loss has reached 0.\033[0m")
        break

    # Evaluate the model
    model_1.eval()  # Set model to evaluation mode
    with torch.inference_mode():
        answer_prediction = model_1(testing_q)
        answer_loss = loss_func(answer_prediction, testing_a)
    print(f"\033[92mEpoch: {epoch+1} | Loss: {loss.item():.4f} | Test loss: {answer_loss.item():.4f}\033[0m")

# Denormalize predictions for visualization
answer_prediction = answer_prediction * (y_max - y_min) + y_min
training_a = training_a * (y_max - y_min) + y_min
training_q = training_q * (X_max - X_min) + X_min
testing_a = testing_a * (y_max - y_min) + y_min
testing_q = testing_q * (X_max - X_min) + X_min

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

    plt.title("Train-Test Split Visualization", fontsize=16)
    plt.xlabel("X values", fontsize=14)
    plt.ylabel("y values", fontsize=14)
    plt.legend(prop={"size": 14})
    plt.grid(True)
    plt.show()

# Visualize predictions
plot_pred(pred=answer_prediction)

