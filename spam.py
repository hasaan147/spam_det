import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk
import json

# Load the model
class SpamCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_filters, filter_sizes, output_size, dropout=0.5):
        super(SpamCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_size)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add channel dimension
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# Initialize model
embed_size = 128
num_filters = 100
filter_sizes = [3, 4, 5]
output_size = 1
dropout = 0.5

# Load vocabulary and max_len
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

vocab_size = len(vocab) + 1

with open('max_len.txt', 'r') as f:
    max_len = int(f.read().strip())

model = SpamCNN(vocab_size, embed_size, num_filters, filter_sizes, output_size, dropout)
model.load_state_dict(torch.load("spam_cnn_model.pth"))

# NLTK data download
nltk.download('punkt')
nltk.download('stopwords')

# Text preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def predict_message(model, message, vocab, max_len):
    model.eval()
    
    # Preprocess the message
    tokens = preprocess_text(message)
    encoded_message = [vocab.get(word, 0) for word in tokens]
    padded_message = encoded_message + [0] * (max_len - len(encoded_message))
    
    # Convert to tensor
    input_tensor = torch.tensor(padded_message).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).item()
        
    return "spam" if prediction > 0.5 else "ham"

# Streamlit app
st.title("Spam Detection using CNN")

st.write("### Enter a message to check if it's spam or ham:")

user_input = st.text_area("Message:")

if st.button("Predict"):
    if user_input:
        prediction = predict_message(model, user_input, vocab, max_len)
        st.markdown(f"<h2 style='text-align: center;'>The message is: <b>{prediction.upper()}</b></h2>", unsafe_allow_html=True)
    else:
        st.write("Please enter a message.")
