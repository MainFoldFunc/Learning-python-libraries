import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Custom dataset #
samples = 1000
input_data, perfect_output = make_circles(samples, noise=0.03, random_state=42)

# Plot the dataset #
plt.scatter(x=input_data[:, 0],
            y=input_data[:, 1],
            c=perfect_output,
            cmap=plt.cm.RdYlBu)
plt.show()

# Turning data to tensors #
input_data = torch.from_numpy(input_data).type(torch.float)
perfect_output = torch.from_numpy(perfect_output).type(torch.float)

# Train-test split #
training_q, testing_q, training_a, testing_a = train_test_split(
    input_data,
    perfect_output,
    test_size=0.2,
    random_state=42
)

# Setting device #
device = "cuda" if torch.cuda.is_available() else "cpu"

# Construct a model #
model_3 = nn.Sequential(
    nn.Linear(in_features=2, out_features=8),
    nn.Linear(in_features=8, out_features=1)
).to(device)

# Predictions (untrained model) #
with torch.no_grad():
    untrained_pred = model_3(training_q.to(device))

print(f"Predictions of untrained model: \n{torch.round(untrained_pred[:10])}")
print(f"What should it be: \n{training_a[:10]}")

# Loss function and optimizer #
loss_func = nn.BCEWithLogitsLoss()
opt_func = torch.optim.SGD(params=model_3.parameters(),
                           lr=0.01)

# Accuracy function #
def acc(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# Predictions on training and testing datasets #
with torch.inference_mode():
    # Training data predictions
    train_logits = model_3(training_q.to(device))
    train_pred_prob = torch.sigmoid(train_logits)
    train_pred = torch.round(train_pred_prob).squeeze()

    # Testing data predictions
    test_logits = model_3(testing_q.to(device))
    test_pred_prob = torch.sigmoid(test_logits)
    test_pred = torch.round(test_pred_prob).squeeze()

# Compare predictions with ground truth #
train_comparison = torch.eq(train_pred, training_a.to(device).squeeze())
test_comparison = torch.eq(test_pred, testing_a.to(device).squeeze())

print(f"Training predictions match ground truth:\n{train_comparison}")
print(f"Testing predictions match ground truth:\n{test_comparison}")

# Calculate and print accuracies #
train_accuracy = acc(training_a.to(device), train_pred)
test_accuracy = acc(testing_a.to(device), test_pred)

print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")

    
