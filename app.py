import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

# Load the trained model
model = joblib.load('spam_email_detector_model.pkl')  # Replace with your actual model file path

# Function to preprocess input text
def preprocess_text(text):
    # Implement your text preprocessing here
    return text

# Streamlit app layout
st.title('Email Spam Detector')
user_input = st.text_area("Enter an email text:", "")

if st.button("Predict"):
    # Preprocess the input
    processed_input = preprocess_text(user_input)
    
    # Vectorize the input
    vectorizer = TfidfVectorizer()
    processed_input_vectorized = vectorizer.transform([processed_input])

    # Make prediction
    prediction = model.predict(processed_input_vectorized)[0]

    # Display prediction
    if prediction == 'ham':
        st.success('This email is not spam!')
    else:
        st.error('This email is spam!')
