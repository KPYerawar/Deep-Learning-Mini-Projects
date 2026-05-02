from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np

app = FastAPI()

# Input schema
class InputText(BaseModel):
    text: str


# Load checkpoint
checkpoint = torch.load("shakespeare_lstm.pt", map_location="cpu")

word_to_idx = checkpoint['word_to_idx']
idx_to_word = checkpoint['idx_to_word']
seq_length = checkpoint['seq_length']


class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, 200)

        self.lstm = nn.LSTM(
            input_size=200,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, vocab_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]

        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# Load model
model = LSTMModel(len(word_to_idx))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


@app.get("/")
def home():
    return {"message": "API Running 🚀"}


# 🔥 Improved prediction
def predict_next_word(text, temperature=0.8):
    words = text.lower().split()[-seq_length:]

    seq = [word_to_idx.get(w, 0) for w in words]

    # 🔹 Padding
    if len(seq) < seq_length:
        seq = [0] * (seq_length - len(seq)) + seq

    seq = torch.tensor(seq).unsqueeze(0)

    with torch.no_grad():
        output = model(seq)

        probs = torch.softmax(output / temperature, dim=1).numpy().flatten()

    pred = np.random.choice(len(probs), p=probs)

    return idx_to_word[pred]

def generate_text(seed, next_words=15, temperature=0.8):
    result = seed

    for _ in range(next_words):
        next_word = predict_next_word(result, temperature)
        result += " " + next_word

    return result

@app.post("/predict") # Endpoint to predict the next word given an input text sequence
def predict(input: InputText):
    next_word = predict_next_word(input.text)
    return {"next_word": next_word}

@app.post("/generate") # New endpoint to generate text based on input seed
def generate(input: InputText):
    generated = generate_text(input.text, next_words=15)
    return {"generated_text": generated}