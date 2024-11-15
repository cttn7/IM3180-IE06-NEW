from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import numpy as np

# Load preprocessed data
df = pd.read_csv('data/processed_data.csv')
documents = df['Document'].tolist()

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Generate embeddings
def generate_embeddings(text_list):
    embeddings = []
    for text in text_list:
        tokens = tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**tokens)
        embeddings.append(output.last_hidden_state.mean(dim=1).squeeze().numpy())  # Mean pooling
    return np.array(embeddings)

print("Generating embeddings...")
embeddings = generate_embeddings(documents)

# Save embeddings and labels
np.save('data/embeddings.npy', embeddings)
np.save('data/labels.npy', df['Label'].to_numpy())
print("Embeddings and labels saved.")
