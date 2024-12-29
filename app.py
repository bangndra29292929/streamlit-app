
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('model.pkl')  # Pastikan model.pkl ada di folder yang sama

# Title of the app
st.title("Prediksi Model Machine Learning")

# Input features
st.write("Masukkan fitur untuk prediksi:")
feature1 = st.number_input("Fitur 1")
feature2 = st.number_input("Fitur 2")
feature3 = st.number_input("Fitur 3")

# Button to make prediction
if st.button("Prediksi"):
    features = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(features)
    st.write(f"Hasil Prediksi: {prediction[0]}")
