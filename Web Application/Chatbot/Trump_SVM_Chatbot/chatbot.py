from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pickle

# Load BERT model and tokenizer for embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load the trained SVM model
with open('data/svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Function to generate embedding for a single comment
def generate_embedding(text):
    tokens = tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy().reshape(1, -1)

# Function to predict sarcasm
def predict_sarcasm(comment):
    embedding = generate_embedding(comment)
    prediction = svm_model.predict(embedding)
    probability = svm_model.predict_proba(embedding)
    return "Sarcastic" if prediction[0] == 1 else "Not Sarcastic", probability
