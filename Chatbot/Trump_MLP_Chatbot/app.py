import streamlit as st
from chatbot import predict_sarcasm

# Streamlit UI setup
st.title("Sarcasm Detection Chatbot")
st.write("Enter a comment below, and the chatbot will tell you if it is sarcastic or not.")

# Input text for sarcasm detection
user_input = st.text_input("Enter a comment to analyze:")

# Button to analyze the input for sarcasm
if st.button("Check Sarcasm"):
    if user_input.strip():
        with st.spinner("Analyzing sarcasm..."):
            result = predict_sarcasm(user_input)
            st.write(f"Prediction: {result}")
