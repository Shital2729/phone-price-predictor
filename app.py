import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Load the Trained Model and Scaler ---
# We load our saved model and scaler to make predictions
try:
    model = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Model or scaler not found. Please run the `train_model.py` script first.")
    st.stop() # Stop the app if model files are missing

# --- Page Configuration ---
st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="centered")

# --- Application Title and Description ---
st.title("📱 Mobile Price Predictor")
st.write(
    "This app predicts the price range of a mobile phone based on its specifications. "
    "Fill in the details below and click 'Predict' to see the result."
)

# --- Create Columns for Input Fields ---
# This helps organize the layout
col1, col2 = st.columns(2)

# --- Input Fields for Mobile Features ---
with col1:
    battery_power = st.number_input('Battery Power (mAh)', min_value=500, max_value=2000, value=1000, step=1)
    blue = st.selectbox('Bluetooth', ('No', 'Yes'))
    clock_speed = st.number_input('Clock Speed (GHz)', min_value=0.5, max_value=3.5, value=1.5, step=0.1)
    dual_sim = st.selectbox('Dual SIM', ('No', 'Yes'))
    fc = st.number_input('Front Camera (MP)', min_value=0, max_value=20, value=5, step=1)
    four_g = st.selectbox('4G', ('No', 'Yes'))
    int_memory = st.number_input('Internal Memory (GB)', min_value=2, max_value=512, value=32, step=1)
    m_dep = st.number_input('Mobile Depth (cm)', min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    mobile_wt = st.number_input('Mobile Weight (gm)', min_value=80, max_value=200, value=140, step=1)
    n_cores = st.number_input('Number of Cores', min_value=1, max_value=8, value=4, step=1)

with col2:
    pc = st.number_input('Primary Camera (MP)', min_value=0, max_value=64, value=12, step=1)
    px_height = st.number_input('Pixel Resolution Height', min_value=0, max_value=2000, value=800, step=1)
    px_width = st.number_input('Pixel Resolution Width', min_value=500, max_value=2000, value=1200, step=1)
    ram = st.number_input('RAM (MB)', min_value=256, max_value=8192, value=2048, step=64)
    sc_h = st.number_input('Screen Height (cm)', min_value=5, max_value=20, value=15, step=1)
    sc_w = st.number_input('Screen Width (cm)', min_value=0, max_value=10, value=7, step=1)
    talk_time = st.number_input('Talk Time (hours)', min_value=2, max_value=20, value=10, step=1)
    three_g = st.selectbox('3G', ('No', 'Yes'))
    touch_screen = st.selectbox('Touch Screen', ('No', 'Yes'))
    wifi = st.selectbox('WiFi', ('No', 'Yes'))

# --- Prediction Logic ---
if st.button('Predict Price Range', key='predict_button'):
    # Convert categorical inputs to numerical (0 or 1)
    blue_val = 1 if blue == 'Yes' else 0
    dual_sim_val = 1 if dual_sim == 'Yes' else 0
    four_g_val = 1 if four_g == 'Yes' else 0
    three_g_val = 1 if three_g == 'Yes' else 0
    touch_screen_val = 1 if touch_screen == 'Yes' else 0
    wifi_val = 1 if wifi == 'Yes' else 0

    # Create the feature list in the correct order for the model
    features = [
        battery_power, blue_val, clock_speed, dual_sim_val, fc, four_g_val,
        int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width,
        ram, sc_h, sc_w, talk_time, three_g_val, touch_screen_val, wifi_val
    ]

    # Scale the features and make a prediction
    features_array = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features_array)
    prediction = model.predict(scaled_features)

    # Map prediction to human-readable output
    price_mapping = {
        0: ("Low Cost", "Under ₹10,000"),
        1: ("Medium Cost", "₹10,000 - ₹20,000"),
        2: ("High Cost", "₹20,000 - ₹40,000"),
        3: ("Very High Cost", "Above ₹40,000")
    }
    
    result_label, result_inr = price_mapping[prediction[0]]

    # --- Display the Result ---
    st.success(f"**Predicted Price Category:** {result_label}")
    st.info(f"**Estimated Price in India:** {result_inr}")