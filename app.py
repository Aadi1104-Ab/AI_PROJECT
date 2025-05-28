import streamlit as st
import pickle
import numpy as np
import os
from sklearn.linear_model import LinearRegression

# === MUST be first Streamlit command ===
st.set_page_config(page_title="AI Sales Predictor", page_icon="🛒", layout="centered", initial_sidebar_state="collapsed")

MODEL_FILENAME = 'sales_forecast_model (1).pkl'

def is_model_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return callable(getattr(obj, 'predict', None))  # checks if obj has predict method
    except Exception:
        return False

# Check if model file exists and is valid; if not, train & save dummy model
if not os.path.exists(MODEL_FILENAME) or not is_model_pickle(MODEL_FILENAME):
    st.write(f"Model file '{MODEL_FILENAME}' missing or invalid. Training dummy model...")
    X_train = np.random.rand(100, 9) * 10
    y_train = X_train[:, 0] * 2 + X_train[:, 1] * 0.5 + np.random.rand(100)
    dummy_model = LinearRegression()
    dummy_model.fit(X_train, y_train)
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump(dummy_model, f)
    st.write("Dummy model trained and saved!")

# Load the model
with open(MODEL_FILENAME, 'rb') as file:
    model = pickle.load(file)
    st.write(f"Loaded model: {type(model)}")

# App UI
st.title("🛒 AI Sales Predictor")
st.markdown("Enter product and date details to predict weekly sales.")

sku_id = st.number_input("SKU ID", min_value=1, step=1)
base_price = st.number_input("Base Price", min_value=0.0, step=0.01)
is_display = st.selectbox("Is Display SKU?", [0, 1])
is_featured = st.selectbox("Is Featured SKU?", [0, 1])
day = st.slider("Day", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)
week = st.slider("Week", 1, 52, 26)
year = st.number_input("Year", min_value=2000, max_value=2030, step=1, value=2025)
store_id = st.number_input("Store ID", min_value=1, step=1)

if st.button("🔮 Predict Sales"):
    try:
        input_features = np.array([[sku_id, base_price, is_display, is_featured, day, month, week, year, store_id]])
        prediction = model.predict(input_features)
        st.success(f"✅ **Predicted Weekly Sales:** {prediction[0]:.2f} units")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
