import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load('disease_outcome_model.joblib')  # Make sure this model file is in the same directory

st.title("Disease Outcome Prediction App")
st.write("Enter your symptoms and demographic details below to predict the disease outcome.")

# Collect user input
age = st.number_input("Age", min_value=0, max_value=120, value=25)
gender = st.selectbox("Gender", options=["Female", "Male"])
fever = st.selectbox("Do you have a fever?", options=["Yes", "No"])
cough = st.selectbox("Do you have a cough?", options=["Yes", "No"])
fatigue = st.selectbox("Do you feel fatigued?", options=["Yes", "No"])
difficulty_breathing = st.selectbox("Do you have difficulty breathing?", options=["Yes", "No"])

# Encode user input
gender_encoded = 1 if gender == "Male" else 0
fever_encoded = 1 if fever == "Yes" else 0
cough_encoded = 1 if cough == "Yes" else 0
fatigue_encoded = 1 if fatigue == "Yes" else 0
difficulty_breathing_encoded = 1 if difficulty_breathing == "Yes" else 0

# Prepare input data for prediction
input_data = np.array([[fever_encoded, cough_encoded, fatigue_encoded, difficulty_breathing_encoded, age, gender_encoded]])

# Make prediction when the button is clicked
if st.button("Predict Outcome"):
    prediction = model.predict(input_data)
    outcome = "Positive" if prediction[0] == 1 else "Negative"
    st.write(f"### Predicted Disease Outcome: {outcome}")
