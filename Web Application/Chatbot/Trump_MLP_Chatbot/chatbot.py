import torch
import torch.nn as nn  # Importing torch.nn for MLP class
import numpy as np
from transformers import BertTokenizer, BertModel

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')

# Load trained MLP model
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

input_size = 768  # Embedding size
hidden_size = 10
output_size = 2  # Binary classification
mlp_model = MLP(input_size, hidden_size, output_size)
mlp_model.load_state_dict(torch.load('data/mlp_model.pth'))
mlp_model.eval()

# Function to generate embeddings for a comment
def generate_embedding(text):
    tokens = tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model_bert(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy().reshape(1, -1)

# Function to predict sarcasm
def predict_sarcasm(comment):
    embedding = generate_embedding(comment)
    embedding_tensor = torch.FloatTensor(embedding)
    with torch.no_grad():
        outputs = mlp_model(embedding_tensor)
        _, predicted_class = torch.max(outputs, 1)
    return "Sarcastic" if predicted_class.item() == 1 else "Not Sarcastic"
