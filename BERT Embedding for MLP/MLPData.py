#importing required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the data
information = torch.load('BERTMeanToken3-5000.pt')
data = information['content']
label = information['sarcasm']

# Check for comment_id and parent_id presence
comment_id = information.get('comment_id', None)
parent_id = information.get('parent_id', None)

# Convert label to integer format if necessary
if isinstance(label, torch.Tensor): 
    label = label.tolist()

# Remove any invalid entries
print("Unique labels before cleaning:", set(label))
label = [int(l) for l in label if isinstance(l, (int, float)) and not pd.isna(l)]

# Convert labels to LongTensor
y = torch.LongTensor(label)

# Check the shape of data
print(f"Original data shape: {data.shape}")
print(f"Number of labels: {len(label)}")

# Embedding size check (e.g., 768 for BERT)
expected_embedding_dim = 768

# Check if the total number of data points matches the expected embedding size
total_data_points = data.shape[0]
if total_data_points % expected_embedding_dim == 0:
    num_samples_based_on_dim = total_data_points // expected_embedding_dim
    print(f"Based on embedding size {expected_embedding_dim}, we get {num_samples_based_on_dim} samples.")

    if num_samples_based_on_dim == len(label):
        print(f"Reshaping data with embedding size {expected_embedding_dim}")
        X = data.view(len(label), expected_embedding_dim)
    elif num_samples_based_on_dim > len(label):
        print(f"Trimming extra {num_samples_based_on_dim - len(label)} samples from data.")
        X = data.view(num_samples_based_on_dim, expected_embedding_dim)[:len(label)]
    else:
        raise ValueError(f"Mismatch: We have {num_samples_based_on_dim} samples based on the embedding size, but expected {len(label)}.")
else:
    raise ValueError(f"Data cannot be evenly divided by the embedding dimension {expected_embedding_dim}. Check the data format.")

# Ensure the number of samples matches the number of labels
if X.shape[0] != len(y):
    raise ValueError(f"The number of samples ({X.shape[0]}) does not match the number of labels ({len(y)})")

# Split data into training and testing sets
if comment_id is not None and parent_id is not None:
    X_train, X_test, y_train, y_test, comment_id_train, comment_id_test, parent_id_train, parent_id_test = train_test_split(
        X, y, comment_id, parent_id, test_size=0.2, random_state=42
    )
else:
    # If comment_id and parent_id are missing, only split X and y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    comment_id_test = parent_id_test = None  # Placeholder for compatibility later on

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

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert scaled data to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Define model parameters
input_size = X_train.shape[1]
hidden_size = 10
output_size = len(torch.unique(y_train))
num_epochs = 100
learning_rate = 0.001

# Create MLP model
model = MLP(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Predict on the test set and store results
model.eval()
with torch.no_grad():
    predicted = model(X_test)
    _, predicted_classes = torch.max(predicted, 1)

# Accuracy calculation
accuracy = (predicted_classes == y_test).sum().item() / y_test.size(0)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
classification_report_str = classification_report(y_test.numpy(), predicted_classes.numpy())
print(classification_report_str)

# Create a DataFrame for results
if comment_id_test is not None and parent_id_test is not None:
    # Include comment_id and parent_id if they exist
    results_df = pd.DataFrame({
        'comment_id': comment_id_test,
        'parent_id': parent_id_test,
        'Actual': y_test.numpy(),
        'Predicted': predicted_classes.numpy()
    })
else:
    # Only include actual and predicted labels if comment_id and parent_id are missing
    results_df = pd.DataFrame({
        'Actual': y_test.numpy(),
        'Predicted': predicted_classes.numpy()
    })

# Now, read the saved CSV file
df = pd.read_csv('ntubus.csv')

# Write both the model prediction results and CSV content into the same Excel file
with pd.ExcelWriter('model_predictions_with_ids2.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='Model Predictions', index=False)
    # Add the content from the CSV file as another sheet
    df.to_excel(writer, sheet_name='CSV Data', index=False)

print("Predictions and actual labels, along with comment_id and parent_id (if available), have been saved to 'model_predictions_with_ids.xlsx'.")
