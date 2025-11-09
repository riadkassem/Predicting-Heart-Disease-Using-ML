import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load the trained pipeline
# -----------------------------
pipe = joblib.load("heart_disease_pipeline.joblib")  # pipeline with preprocessor + XGBClassifier

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict the risk of heart disease")

# Numeric features
NUM_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
NUM_RANGES = {
    'age': (29, 77),
    'trestbps': (94, 200),
    'chol': (126, 564),
    'thalach': (71, 202),
    'oldpeak': (0.0, 6.2),
    'ca': (0, 4)
}

inputs = {}
for feature in NUM_FEATURES:
    min_val, max_val = NUM_RANGES[feature]
    default = (min_val + max_val) // 2
    inputs[feature] = st.number_input(
        feature,
        min_value=min_val,
        max_value=max_val,
        value=default
    )

# Categorical features
inputs['sex'] = st.selectbox("Sex", ("Male", "Female"))
inputs['sex'] = 1 if inputs['sex'] == "Male" else 0

inputs['cp'] = st.selectbox("Chest Pain Type (0-3)", (0, 1, 2, 3))
inputs['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1))
inputs['restecg'] = st.selectbox("Resting ECG (0-2)", (0, 1, 2))
inputs['exang'] = st.selectbox("Exercise Induced Angina", (0, 1))
inputs['slope'] = st.selectbox("Slope (0-2)", (0, 1, 2))
inputs['thal'] = st.selectbox("Thalassemia (0-3)", (0, 1, 2, 3))

# -----------------------------
# Prepare input DataFrame
# -----------------------------
features = pd.DataFrame([inputs])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    prediction = pipe.predict(features)[0]
    probability = pipe.predict_proba(features)[0][prediction] * 100

    if prediction == 1:
        st.error(f"🚨 The model predicts: Heart Disease ({probability:.1f}% confidence)")
    else:
        st.success(f"✅ The model predicts: No Heart Disease ({probability:.1f}% confidence)")
