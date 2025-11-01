# Smart Elderly Care Dashboard with ML Risk Prediction
# Professional Medical Theme Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Elderly Care Dashboard", page_icon="🩺", layout="wide")

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv('elderly_health_dataset.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966481.png", width=100)
st.sidebar.title("Smart Elderly Care Dashboard")
st.sidebar.markdown("Monitor, Analyze, and Predict Health Risks in Real-Time")

# --- Filters ---
selected_elderly = st.sidebar.selectbox("Select Elderly ID", df['elderly_id'].unique())
selected_feature = st.sidebar.selectbox("Select Metric", ['heart_rate', 'temperature', 'spo2'])

# --- Main Dashboard ---
st.title("🩺 Smart Elderly Care Dashboard with ML Risk Prediction")

col1, col2, col3 = st.columns(3)

latest = df[df['elderly_id'] == selected_elderly].iloc[-1]
col1.metric("Heart Rate (bpm)", latest['heart_rate'])
col2.metric("Temperature (°C)", latest['temperature'])
col3.metric("SpO₂ (%)", latest['spo2'])

# --- Chart ---
st.subheader(f"{selected_feature.replace('_', ' ').title()} Trend for {selected_elderly}")
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(df[df['elderly_id'] == selected_elderly]['timestamp'], df[df['elderly_id'] == selected_elderly][selected_feature], label=selected_feature, linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel(selected_feature)
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Load Model ---
try:
    with open('risk_model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success("ML Model Loaded Successfully")
except:
    st.warning("⚠️ ML model not found. Please train the model first in Colab.")

# --- Risk Prediction ---
st.subheader("🧠 Risk Prediction")
input_data = df[df['elderly_id'] == selected_elderly].iloc[-1][['heart_rate', 'spo2', 'temperature', 'motion', 'fall_detected', 'blood_pressure_sys', 'blood_pressure_dia', 'ambient_temp', 'humidity']].values.reshape(1, -1)

if 'model' in locals():
    prediction = model.predict(input_data)[0]
    st.markdown(f"### 🩸 Predicted Health Status: **{prediction}**")
    if prediction == "Critical":
        st.error("Immediate medical attention required! 🚨")
    elif prediction == "At-Risk":
        st.warning("Monitor closely. Abnormal readings detected.")
    else:
        st.success("Stable condition. No anomaly detected.")

# --- Data Summary ---
st.subheader("📈 Dataset Overview")
st.dataframe(df.head(10))
st.caption(f"Total Records: {len(df)}")
