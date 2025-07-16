
import streamlit as st
import pandas as pd
import joblib
import urllib.request

# Load model from GitHub
MODEL_URL = "https://raw.githubusercontent.com/gurpinder7473/water-quality-ai/main/best_water_model.pkl"
urllib.request.urlretrieve(MODEL_URL, "best_water_model.pkl")
model = joblib.load("best_water_model.pkl")

st.title("💧 AquaMind: Water Quality Predictor")
st.markdown("Upload water data or enter values manually to predict water safety.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded data:")
    st.write(data.head())

    if st.button("Predict"):
        result = model.predict(data)
        st.success("✅ Predictions done!")
        st.write(result)
else:
    st.markdown("Or enter data manually:")

    f1 = st.number_input("pH")
    f2 = st.number_input("Hardness")
    f3 = st.number_input("Solids")
    f4 = st.number_input("Chloramines")
    f5 = st.number_input("Sulfate")
    f6 = st.number_input("Conductivity")
    f7 = st.number_input("Organic Carbon")
    f8 = st.number_input("Trihalomethanes")
    f9 = st.number_input("Turbidity")

    input_data = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

    if st.button("Predict Quality"):
        pred = model.predict([input_data])
        st.success("💧 Water is Safe" if pred[0] == 1 else "❌ Water is Not Safe")
