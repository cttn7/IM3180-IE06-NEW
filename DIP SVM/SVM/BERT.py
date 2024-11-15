import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

# Use Hugging Face's model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
model = BertModel.from_pretrained('bert-base-uncased')

# Load the Excel file
df = pd.read_excel('Trump_2-201.xlsx')

# Extract relevant columns
content = df['comment_body'].tolist()
user = df['comment_author'].tolist()
score = [str(s) for s in df['comment_score'].tolist()]  # Convert score to string
sarcasm = ["sarcastic" if s == 1 else "not_sarcastic" for s in df['OPR_b/hy'].tolist()]  # Binary to text
view = ["positive" if v == 1 else "negative" for v in df['pos/neg view_b/hy'].tolist()]  # Binary to text

# Function for last token embedding
def lastToken_embedding(variate):
    inputs = tokenizer(variate, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    last_token_embedding = outputs.last_hidden_state[:, -1, :]
    return last_token_embedding

# Function for mean embedding
def mean_embedding(variate):
    inputs = tokenizer(variate, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use mean pooling to get the representation of the entire sequence
    pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
    return pooled_output.squeeze()

# Calculate embeddings
lastToken_embedding_user = lastToken_embedding(user)
lastToken_embedding_content = lastToken_embedding(content)
print("lastToken_embedding_user:", lastToken_embedding_user.shape)
print("lastToken_embedding_content:", lastToken_embedding_content.shape)

mean_embedding_user = mean_embedding(user)
mean_embedding_content = mean_embedding(content)
mean_embedding_score = mean_embedding(score)  # Should now work
mean_embedding_sarcasm = mean_embedding(sarcasm)  # Should now work
mean_embedding_view = mean_embedding(view)  # Should now work

print("mean_embedding_user: ", mean_embedding_user.shape)
print("mean_embedding_content: ", mean_embedding_content.shape)
print("Mean embedding score:", mean_embedding_score)
print("Mean embedding sarcasm:", mean_embedding_sarcasm)
print("Mean embedding view:", mean_embedding_view)

# Save each embedding separately
torch.save(mean_embedding_user, 'mean_embedding_user.pt')
torch.save(mean_embedding_content, 'mean_embedding_content.pt')
torch.save(mean_embedding_score, 'mean_embedding_score.pt')
torch.save(mean_embedding_sarcasm, 'mean_embedding_sarcasm.pt')
torch.save(mean_embedding_view, 'mean_embedding_view.pt')