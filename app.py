import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("heart_disease_data.csv")

# Prepare data
X = df.drop(columns='target', axis=1)
y = df['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Streamlit UI
st.title("Heart Disease Prediction App")
st.write("Enter patient details to predict the likelihood of heart disease.")

# Dropdown options
sex_options = {0: "Female", 1: "Male"}
cp_options = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
fbs_options = {0: "False", 1: "True"}
restecg_options = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
exang_options = {0: "No", 1: "Yes"}
slope_options = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
ca_options = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}
thal_options = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=sex_options.keys(), format_func=lambda x: sex_options[x])
cp = st.selectbox("Chest Pain Type", options=cp_options.keys(), format_func=lambda x: cp_options[x])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=fbs_options.keys(), format_func=lambda x: fbs_options[x])
restecg = st.selectbox("Resting ECG", options=restecg_options.keys(), format_func=lambda x: restecg_options[x])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise-Induced Angina", options=exang_options.keys(), format_func=lambda x: exang_options[x])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", options=slope_options.keys(), format_func=lambda x: slope_options[x])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=ca_options.keys(), format_func=lambda x: ca_options[x])
thal = st.selectbox("Thalassemia Type", options=thal_options.keys(), format_func=lambda x: thal_options[x])

# Make prediction
input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
    st.subheader(f"Result: {result}")
