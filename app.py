import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the trained scaler from file
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the trained SVM model from file
with open('svm_classifier.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Function to predict the class for new data
def predict_class(input_data):
    # Preprocess the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data.reshape(1, -1)) # Use the loaded 'scaler' object 
    # Predict the class label using the loaded SVM model
    prediction = svm_model.predict(input_data_scaled)
    return prediction

# Streamlit app
def main():
    st.title('SVM Classifier Deployment')
    st.write('This app predicts the nutritional status based on anthropometric data.')

    # Sidebar with user input fields
    st.sidebar.header('User Input Parameters')
    def user_input_features():
        age = st.sidebar.slider('Age', 1, 18, 10)
        weight = st.sidebar.slider('Weight (kg)', 0.0, 150.0, 30.0)
        height = st.sidebar.slider('Height (cm)', 50.0, 200.0, 100.0)
        gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
        gender_code = 1 if gender == 'Male' else 0  # Encode gender as 1 for Male, 0 for Female
        return np.array([age, weight, height, gender_code])

    input_data = user_input_features()

    # Display user input data
    st.subheader('User Input:')
    st.write('Age:', input_data[0])
    st.write('Weight:', input_data[1], 'kg')
    st.write('Height:', input_data[2], 'cm')
    st.write('Gender:', 'Male' if input_data[3] == 1 else 'Female')

    # Predicting the class label
    prediction = predict_class(input_data)  # Include all features for prediction

    # Debugging statements
    st.write('Input Data:', input_data)
    st.write('Prediction:', prediction)

    # Mapping prediction to result category
    if prediction == 1:
        result = 'Gizi Baik'
    elif prediction == 2:
        result = 'Gizi Kurang => Stunting'
    elif prediction == 3:
        result = 'Risiko Gizi Lebih'
    elif prediction == 4:
        result = 'Gizi Lebih'
    else:
        result = 'Unknown'  # Handle any unexpected prediction

    # Display the prediction result
    st.success(f"Nutritional Status Prediction: {result}")
