import streamlit as st
import pandas as pd
import joblib
from ml.utils import transform_amount_with_scaler

st.title("💳 Credit Card Fraud Detection (Streamlit Demo)")

model = joblib.load('models/xgb_model.joblib')
scaler = joblib.load('models/scaler.joblib')

uploaded = st.file_uploader("Upload CSV (Time, Amount, V1–V28)", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    df = transform_amount_with_scaler(df, scaler)
    preds = model.predict_proba(df)[:, 1]
    df['fraud_probability'] = preds
    st.dataframe(df.head(10))
    st.download_button("Download Results", df.to_csv(index=False), "fraud_predictions.csv")

st.write("---")
st.write("Or enter single transaction:")
input_data = { 'Time': st.number_input('Time', 0), 'Amount': st.number_input('Amount', 0.0) }
for i in range(1, 29):
    input_data[f'V{i}'] = st.number_input(f'V{i}', 0.0)
if st.button("Predict"):
    X = pd.DataFrame([input_data])
    X = transform_amount_with_scaler(X, scaler)
    prob = model.predict_proba(X)[0, 1]
    st.success(f"Fraud probability: {prob:.4f}")
