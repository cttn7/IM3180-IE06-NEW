import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.utils import resample
import torch

# Load the data and labels
data = torch.load('mean_embedding_content.pt', weights_only=True)
label = torch.load('mean_embedding_sarcasm.pt', weights_only=True)
# Load the original Excel file containing comments and other data
original_df = pd.read_excel('Trump_2-201.xlsx')  # Update with actual file path and name

# Print the shape to understand what you're working with
print(f"Loaded label shape: {label.shape}")

# Prepare label data
if len(label.shape) > 1:
    label = label[:, 0]  # Adjust according to the structure of your label data

label = label.numpy()  # Convert to numpy if necessary

# Encode text labels to numerical labels
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(label)  # Encode 'sarcastic' and 'not_sarcastic'

# Split the data into minority and majority classes
minority_indices = np.where(label_encoded == 1)[0]
majority_indices = np.where(label_encoded == 0)[0]

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
test_comments = original_df.iloc[test_idx]['comment_body']  # Adjust 'Comment' to the actual column name

# Create DataFrame for test results
test_results = pd.DataFrame({
    'Original Comment': test_comments.values,   # Original comments from Excel
    'True Label': y_test.numpy(),
    'Predicted Label': y_pred
})

# Print the first few rows to inspect
print(test_results.head())

# Save to an Excel file
test_results.to_excel('labeled_test_dataset_with_comments.xlsx', index=False)
print("Labeled test dataset with original comments saved to 'labeled_test_dataset_with_comments.xlsx'")