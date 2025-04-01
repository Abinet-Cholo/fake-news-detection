import streamlit as st
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load Model and Tokenizer ---
@st.cache_resource
def load_resources():
    """
    Loads the pre-trained LSTM model and tokenizer from the local directory.
    Adjust the paths if necessary based on your repository structure.
    """
    model_path = os.path.join(os.getcwd(), 'lstm_fake_news_model.h5')
    tokenizer_path = os.path.join(os.getcwd(), 'tokenizer.pkl')
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_resources()

# --- Constants ---
MAX_LENGTH = 500  # Must match the max_length used during training

# --- Text Cleaning Function ---
def clean_text(text: str) -> str:
    """
    Converts text to lowercase and removes non-alphabetic characters (except spaces).
    """
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c == ' '])
    return text

# --- Prediction Function ---
def predict_news(news_header: str) -> float:
    """
    Cleans and tokenizes the news header, pads the sequence, 
    and returns the model's prediction score.
    """
    cleaned_text = clean_text(news_header)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded_seq = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post')
    prediction = model.predict(padded_seq)
    # prediction is an array of shape (1,1), extract the single value
    return float(prediction[0][0])

# --- Streamlit App Layout ---
st.title("Fake News Detection App")
st.write(
    "Use this LSTM model to check if a news headline is potentially real or fake. "
    "Enter a headline below and click 'Predict' to see the result."
)

# Sidebar with model info
st.sidebar.header("Model Information")
st.sidebar.markdown(
    """
    - **Model:** LSTM for binary classification (Fake vs. Real).
    - **Tokenizer:** Trained on top 5000 words.
    - **Max Sequence Length:** 500 tokens.
    - **Epochs:** 5
    - **Batch Size:** 64
    - **Accuracy:** ~ (Insert your model's accuracy)
    """
)

# Text area for user input
news_header_input = st.text_area("Enter a news headline", "Type your news headline here...")

# Predict button
if st.button("Predict"):
    if news_header_input.strip():
        score = predict_news(news_header_input)
        # If score > 0.5, it's more likely to be Real, otherwise Fake
        if score > 0.5:
            st.success("This news is likely **Real**.")
        else:
            st.warning("This news is likely **Fake**.")
        st.write("**Prediction Score:**", round(score, 4))
    else:
        st.error("Please enter a news headline to get a prediction.")
