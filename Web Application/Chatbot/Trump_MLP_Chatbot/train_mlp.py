import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np

# Load embeddings and labels
embeddings = np.load('data/embeddings.npy')
labels = np.load('data/labels.npy')

# Convert data to tensors
X = torch.FloatTensor(embeddings)
y = torch.LongTensor(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model parameters
input_size = X_train.shape[1]
hidden_size = 10
output_size = len(torch.unique(y_train))
learning_rate = 0.001
num_epochs = 100

# Create model
model = MLP(input_size, hidden_size, output_size)

# Define weighted loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Calculate training accuracy
    with torch.no_grad():
        _, predicted_classes = torch.max(outputs, 1)
        train_accuracy = (predicted_classes == y_train).sum().item() / y_train.size(0)

    # Print epoch loss and accuracy
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {train_accuracy * 100:.2f}%')

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, test_predicted_classes = torch.max(test_outputs, 1)
    test_accuracy = (test_predicted_classes == y_test).sum().item() / y_test.size(0)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'data/mlp_model.pth')
print("MLP model trained and saved.")
