import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict the risk of heart disease")

# --- Input fields ---
age = st.number_input("Age", 1, 120, 40)
sex = st.selectbox("Sex", ("Male", "Female"))
cp = st.selectbox("Chest Pain Type (0-3)", (0, 1, 2, 3))
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1))
restecg = st.selectbox("Resting ECG (0-2)", (0, 1, 2))
thalach = st.number_input("Max Heart Rate Achieved", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina", (0, 1))
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope (0-2)", (0, 1, 2))
ca = st.selectbox("Number of Major Vessels (0-4)", (0, 1, 2, 3, 4))
thal = st.selectbox("Thalassemia (0-3)", (0, 1, 2, 3))

# --- Preprocess inputs ---
# Convert sex to int
sex_val = 1 if sex == "Male" else 0

# Create DataFrame with proper columns and types
features = pd.DataFrame([{
    'age': int(age),
    'sex': int(sex_val),
    'cp': int(cp),
    'trestbps': int(trestbps),
    'chol': int(chol),
    'fbs': int(fbs),
    'restecg': int(restecg),
    'thalach': int(thalach),
    'exang': int(exang),
    'oldpeak': float(oldpeak),
    'slope': int(slope),
    'ca': int(ca),
    'thal': int(thal)
}])

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction] * 100

    if prediction == 1:
        st.error(f"🚨 The model predicts: Heart Disease ({probability:.1f}% confidence)")
    else:
        st.success(f"✅ The model predicts: No Heart Disease ({probability:.1f}% confidence)")
