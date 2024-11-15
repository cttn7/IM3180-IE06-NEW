import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

# Hugging Face tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
# Using BERT model
model = BertModel.from_pretrained('bert-base-uncased')

df = pd.read_csv('ntu_dataset.csv')
content = df['comment_body'].tolist()
user = df['comment_author'].tolist()
sarcasm = df['sarcasm_score'].tolist()
relationship = df['sentiment_score' ].tolist()

def lastToken_embedding(variate):
    embeddings = []
    for text in variate:
        # Check if the text is valid (string and not empty)
        if isinstance(text, str) and text.strip():  # Check for non-empty strings
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            last_token_embedding = outputs.last_hidden_state[:, -1, :]
            embeddings.append(last_token_embedding)
    return torch.cat(embeddings) if embeddings else torch.empty((0, 768))  # Adjust the size if no valid texts

def mean_embedding(variate):
    embeddings = []
    for text in variate:
        # Check if the text is valid (string and not empty)
        if isinstance(text, str) and text.strip():  # Check for non-empty strings
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            # Use mean pooling to obtain representation of the entire sequence
            pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
            embeddings.append(pooled_output.squeeze())
    return torch.cat(embeddings) if embeddings else torch.empty((0, 768))  # Adjust the size if no valid texts

# Embedding 
lastToken_embedding_user = lastToken_embedding(user)
lastToken_embedding_content = lastToken_embedding(content)
print("lastToken_embedding_user:", lastToken_embedding_user)
print("lastToken_embedding_content:", lastToken_embedding_content)

mean_embedding_user = mean_embedding(user)
mean_embedding_content = mean_embedding(content)
print("mean_embedding_user: ", mean_embedding_user)
print("mean_embedding_content: ", mean_embedding_content)

mean_embedding = {
    'content': mean_embedding_content,
    'user_info': mean_embedding_user,
    'sarcasm': sarcasm,
    'relationship': relationship,
}
# Save embedding pytorch file
torch.save(mean_embedding, 'BERTMeanToken3.pt')
