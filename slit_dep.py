import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("knn_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Streamlit app title
st.title("Diabetes Prediction App")
st.write("Enter the medical details to predict the likelihood of diabetes.")

# User input sliders for features (example based on PIMA dataset)
pregnancies = st.slider("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.slider("Glucose Level", min_value=0, max_value=200, step=1)
blood_pressure = st.slider("Blood Pressure (mm Hg)", min_value=0, max_value=140, step=1)
skin_thickness = st.slider("Skin Thickness (mm)", min_value=0, max_value=100, step=1)
bmi = st.slider("BMI", min_value=0.0, max_value=70.0, step=0.1)
dpf = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
age = st.slider("Age", min_value=10, max_value=100, step=1)

# Prediction logic
if st.button("Predict"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,bmi, dpf, age]])
    prediction = model.predict(features)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.success(f"The model predicts: **{result}**")
