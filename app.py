import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('model.pkl')

# Title of the app
st.title("Prediksi Model Machine Learning")

# Input features (4 features for Iris dataset)
st.write("Masukkan fitur untuk prediksi:")
feature1 = st.number_input("Fitur 1", format="%.2f")
feature2 = st.number_input("Fitur 2", format="%.2f")
feature3 = st.number_input("Fitur 3", format="%.2f")
feature4 = st.number_input("Fitur 4", format="%.2f")  # Tambahkan fitur ke-4

# Button to make prediction
if st.button("Prediksi"):
   features = np.array([[feature1, feature2, feature3, feature4]])  # Pastikan 4 fitur
   try:
       prediction = model.predict(features)
       st.write(f"Hasil Prediksi: {prediction[0]}")
   except Exception as e:
       st.error(f"Terjadi kesalahan: {e}")
