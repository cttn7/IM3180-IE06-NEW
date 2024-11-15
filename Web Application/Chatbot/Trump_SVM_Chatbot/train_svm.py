from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
import pickle

# Load embeddings and labels
embeddings = np.load('data/embeddings.npy')
labels = np.load('data/labels.npy')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Train an SVM model
print("Training SVM...")
svm_model = SVC(kernel='linear', probability=True)  # Linear kernel for classification
svm_model.fit(X_train, y_train)

# Save the trained model
with open('data/svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm_model, model_file)
print("SVM model trained and saved.")

# Evaluate the SVM model
y_pred = svm_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
