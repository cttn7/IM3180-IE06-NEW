import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class SarcasmMLP(nn.Module):
    def __init__(self, input_dim):
        super(SarcasmMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RAGChatbot:
    def __init__(self, sarcasm_model_path="sarcasm_mlp_model.pth"):
        # Load the sarcasm detection model
        self.model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        input_dim = 768  # Dimension of the embeddings
        self.sarcasm_model = SarcasmMLP(input_dim)
        self.sarcasm_model.load_state_dict(torch.load(sarcasm_model_path))
        self.sarcasm_model.eval()  # Set model to evaluation mode

    def detect_sarcasm(self, text):
        # Convert text to embeddings
        text_embedding = self.model.encode([text])
        text_tensor = torch.tensor(text_embedding, dtype=torch.float32)

        # Predict sarcasm
        with torch.no_grad():
            output = self.sarcasm_model(text_tensor)
            print("Model Output:", output)  # Debug: Print model output
            prediction = torch.argmax(output, axis=1).item()
            print("Prediction:", prediction)  # Debug: Print prediction result

        return "Sarcastic" if prediction == 1 else "Not Sarcastic"

    def generate_answer(self, query):
        sarcasm_result = self.detect_sarcasm(query)
        return f"The comment is: {sarcasm_result}"
