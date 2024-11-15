import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.utils import resample
import torch

# Load the .pt file
bert_data = torch.load('BERTMeanToken3.pt', weights_only=True)  # Load the file

# Debug: Inspect the keys in the loaded data
print(f"Keys in BERTMeanToken3.pt: {bert_data.keys()}")

data = bert_data['content']  # Update with the correct key for embeddings
label = bert_data['sarcasm']  # Update with the correct key for labels

# Debugging: Check type and structure of the loaded label
if isinstance(label, list):
    label = np.array(label)  # Convert list to NumPy array
    print("Label converted from list to NumPy array.")
elif isinstance(label, torch.Tensor):
    label = label.numpy()  # Convert PyTorch tensor to NumPy array
    print("Label converted from PyTorch tensor to NumPy array.")

# Confirm label type and shape
print(f"Type of label: {type(label)}")
print(f"Label shape: {label.shape}")

# Load the CSV file containing comments and other data
original_df = pd.read_csv('politics-only-topic-2000.csv')  # Update with actual file path and name

# Split the data into minority and majority classes
minority_indices = np.where(label == 1)[0]
majority_indices = np.where(label == 0)[0]

minority_class = data[minority_indices]
majority_class = data[majority_indices]

# Oversample the minority class
if len(minority_class) > 0 and len(majority_class) > 0:
    minority_class_oversampled = resample(
        minority_class,
        replace=True,     
        n_samples=len(majority_class),  
        random_state=42
    )
    
    # Combine majority and oversampled minority classes
    balanced_data = np.concatenate([majority_class, minority_class_oversampled])
    balanced_labels = np.concatenate([np.zeros(len(majority_class)), np.ones(len(minority_class_oversampled))])
else:
    raise ValueError("Either minority or majority class is empty; cannot balance data.")

# Convert balanced data to tensors
X = torch.tensor(balanced_data)  
y = torch.tensor(balanced_labels).long()

# Split the dataset and keep track of indices
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, y, np.arange(len(X)), test_size=0.2, random_state=30
)

# Ensure data is 2D
X_train = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
X_test = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test

if X_train.ndim == 1:  # Check if data is 1D
    X_train = X_train.reshape(-1, 1)
if X_test.ndim == 1:  # Check if data is 1D
    X_test = X_test.reshape(-1, 1)

print(f"X_train shape before scaling: {X_train.shape}")
print(f"X_test shape before scaling: {X_test.shape}")

# Standardize feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVC model
model = SVC(kernel='rbf', class_weight='balanced')
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred, zero_division=1))

# Retrieve the original comments for the test set based on the test indices
test_comments = original_df.iloc[test_idx]['comment']  # Adjust 'comment_body' to the actual column name

# Create DataFrame for test results
test_results = pd.DataFrame({
    'Original Comment': test_comments.values,   # Original comments from CSV
    'True Label': y_test.numpy(),
    'Predicted Label': y_pred
})

# Print the first few rows to inspect
print(test_results.head())

# Save to an Excel file
test_results.to_excel('labeled_test_dataset_with_comments.xlsx', index=False)
print("Labeled test dataset with original comments saved to 'labeled_test_dataset_with_comments.xlsx'")