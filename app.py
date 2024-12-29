import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('model.pkl')

# Title of the app
st.title("Prediksi Model Machine Learning")

# Input features
st.write("Masukkan fitur untuk prediksi:")
feature1 = st.number_input("Fitur 1", format="%.2f")
feature2 = st.number_input("Fitur 2", format="%.2f")
feature3 = st.number_input("Fitur 3", format="%.2f")

# Button to make prediction
if st.button("Prediksi"):
    # Pastikan semua fitur diisi
    if feature1 is not None and feature2 is not None and feature3 is not None:
        features = np.array([[feature1, feature2, feature3]])
        prediction = model.predict(features)
        st.write(f"Hasil Prediksi: {prediction[0]}")
    else:
        st.error("Silakan masukkan semua fitur.")
