import torch
from pathlib import Path
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import numpy as np

# Custom dataset
samples = 10000
input_data, perfect_output = make_circles(samples, noise=0.01, random_state=20)

# Plot the dataset
#plt.scatter(x=input_data[:, 0],
#            y=input_data[:, 1],
#            c=perfect_output,
#            cmap=plt.cm.RdYlBu)
#plt.show()

# Turning data to tensors
input_data = torch.from_numpy(input_data).type(torch.float32)
perfect_output = torch.from_numpy(perfect_output).type(torch.float32)

# Train-test split
training_q, testing_q, training_a, testing_a = train_test_split(
    input_data,
    perfect_output,
    test_size=0.2,
    random_state=42
)

# Setting device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Construct a model
model_3 = nn.Sequential(
    nn.Linear(in_features=2, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=1)
   
).to(device)

# Loss function and optimizer
loss_func = nn.BCEWithLogitsLoss()
opt_func = torch.optim.SGD(params=model_3.parameters(), lr=0.01)

# Accuracy function
def acc(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


# Function to plot decision boundary with automatic closing of previous plot
def plot_decision_boundary(model, X, y, epoch):
    plt.clf()  # Clear the previous plot

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        predictions = torch.sigmoid(model(grid)).reshape(xx.shape)
    
    plt.contourf(xx, yy, predictions.cpu().numpy(), alpha=0.7, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor="white")
    plt.title(f"Decision Boundary at Epoch {epoch}")
    
    # Non-blocking plot updates
    plt.draw()
    plt.pause(0.001)  # Pause for a short time to allow the plot to update

# Training loop parameters
epochs = 10000

# Move data to the device
training_q, testing_q = training_q.to(device), testing_q.to(device)
training_a, testing_a = training_a.to(device), testing_a.to(device)

# Lists to store metrics
loss_values = []
test_accuracies = []

# Training loop
for epoch in range(epochs):
    model_3.train()

    # Forward pass
    logits = model_3(training_q).squeeze()
    loss = loss_func(logits, training_a)
    loss_values.append(loss.item())
    
    # Backward pass
    opt_func.zero_grad()
    loss.backward()
    opt_func.step()

    # Evaluation
    model_3.eval()
    with torch.no_grad():
        # Training accuracy
        train_logits = model_3(training_q).squeeze()
        train_pred = torch.round(torch.sigmoid(train_logits))
        train_accuracy = acc(training_a, train_pred)

        # Testing accuracy
        test_logits = model_3(testing_q).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_accuracy = acc(testing_a, test_pred)
        test_accuracies.append(test_accuracy)

    # Plot decision boundary every 10 epochs
    if (epoch + 1) % 100 == 0:
        plot_decision_boundary(model_3, input_data.cpu().numpy(), perfect_output.cpu().numpy(), epoch + 1)

    # Print metrics
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{epochs} \t Loss: {loss:.4f} \t Train Acc: {train_accuracy:.2f}% \t Test Acc: {test_accuracy:.2f}%")

    if test_accuracy > 97:
        print(f"Early stopping at epoch {epoch + 1}: Test Accuracy > 97%")
        break

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(len(loss_values)), loss_values, label="Loss")  # Use len(loss_values) for the x-axis
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()

# Plot the test accuracy curve
plt.figure(figsize=(10, 6))
plt.plot(range(len(test_accuracies)), test_accuracies, label="Test Accuracy")  # Use len(test_accuracies) for x-axis
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy Curve")
plt.legend()
plt.show()

# Final evaluation
print("\nTraining complete.")
model_3.eval()
with torch.no_grad():
    final_train_logits = model_3(training_q).squeeze()
    final_train_pred = torch.round(torch.sigmoid(final_train_logits))
    final_train_accuracy = acc(training_a, final_train_pred)

    final_test_logits = model_3(testing_q).squeeze()
    final_test_pred = torch.round(torch.sigmoid(final_test_logits))
    final_test_accuracy = acc(testing_a, final_test_pred)

print(f"Final Training Accuracy: {final_train_accuracy:.2f}%")
print(f"Final Testing Accuracy: {final_test_accuracy:.2f}%")

# Save the model
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_DIR / "circle_model.pth"

torch.save(model_3.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}") 

