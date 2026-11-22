import streamlit as st
import pandas as pd
import numpy as np
import joblib
from joblib import load
import os

# --- LOAD MODEL ---
model = load('Heart_Diesease_prediction.joblib')

# --- PAGE SETUP ---
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details below.")

# --- INPUTS IN 3 COLUMNS ---
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=52)
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=125)
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
    oldpeak = st.number_input("Oldpeak", value=1.0)
    ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3, 4])

with col2:
    sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    chol = st.number_input("Cholesterol (chol)", value=212)
    thalach = st.number_input("Max Heart Rate (thalach)", value=168)
    slope = st.selectbox("Slope", [0, 1, 2])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

with col3:
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [0, 1])
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])

# --- PREDICT BUTTON ---
if st.button("Predict Heart Disease", use_container_width=True):
    # Order MUST be same as training: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, tha
    
    input = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
       exang, oldpeak, slope, ca, thal]],
                               columns=[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']])
    
    prediction = model.predict(input)[0]
    probability = model.predict_proba(input)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK: Patient likely has Heart Disease")
        st.write(f"Probability: **{probability:.2f}%**")
        st.write("Advice: Please see a doctor immediately for further checkup.")
    else:
        st.success(f"✅ LOW RISK: Patient likely does NOT have Heart Disease")
        st.write(f"Probability of disease: **{probability:.2f}%**")
        st.write("healthy status: good healthy status")
        st.write("Advice: Keep healthy lifestyle, exercise and check regularly.")
    st.balloons()