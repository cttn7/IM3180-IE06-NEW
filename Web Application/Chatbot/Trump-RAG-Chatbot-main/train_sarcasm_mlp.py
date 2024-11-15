import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Load the data with specified encoding
df = pd.read_csv("data/trump_slow_meltdown.csv", encoding="ISO-8859-1")
comments = df["comment"]
labels = df["OPR"]

# Convert comments to embeddings
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
embeddings = model.encode(comments.tolist(), show_progress_bar=True)
X = np.array(embeddings)
y = np.array(labels)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the MLP model
class SarcasmMLP(nn.Module):
    def __init__(self, input_dim):
        super(SarcasmMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Output layer for 2 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
input_dim = X.shape[1]
sarcasm_model = SarcasmMLP(input_dim)

# Calculate class weights
class_counts = torch.bincount(torch.tensor(y_train))
class_weights = 0.5 / class_counts.float()
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(sarcasm_model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = sarcasm_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
with torch.no_grad():
    predictions = sarcasm_model(X_test_tensor)
    predicted_labels = torch.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_test_tensor, predicted_labels)
    balanced_acc = balanced_accuracy_score(y_test_tensor, predicted_labels)
    f1 = f1_score(y_test_tensor, predicted_labels, average='weighted')

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_acc * 100:.2f}%")
    print(f"F1 Score: {f1:.2f}")

# Save the model
torch.save(sarcasm_model.state_dict(), "sarcasm_mlp_model.pth")
print("Model saved as sarcasm_mlp_model.pth")
