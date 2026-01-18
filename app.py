import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ==================================================
# Load model and feature columns
# ==================================================
model = joblib.load("car_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

CURRENT_YEAR = datetime.now().year

# ==================================================
# Page configuration
# ==================================================
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Prediction System")
st.markdown(
    "Predict **used car prices (USD)** using a machine learning model trained "
    "on Indian market data (prices converted from INR lakhs)."
)

st.divider()

# ==================================================
# Sidebar
# ==================================================
st.sidebar.header("📊 About")
st.sidebar.write(
    """
    **Model:** Random Forest Regressor  
    **Target:** Car Price (USD)  
    **Market:** Indian used-car market  
    **Extras:** Confidence range & market calibration  
    """
)

# ==================================================
# Brand → Model mapping
# ==================================================
brand_model_map = {
    "Maruti": ["Swift", "Baleno", "Alto"],
    "Hyundai": ["i10", "i20", "Creta"],
    "Honda": ["City", "Amaze", "Civic"],
    "Toyota": ["Corolla", "Innova", "Fortuner"],
    "Mercedes-Benz": ["C-Class", "E-Class", "S-Class"],
    "BMW": ["3 Series", "5 Series", "X5"],
    "Audi": ["A4", "A6", "Q7"],
    "Ferrari": ["488 Gtb"],
    "Rolls-Royce": ["Ghost"],
}

# ==================================================
# Smart defaults by brand
# ==================================================
brand_defaults = {
    "Maruti": {"engine": 1200, "power": 82, "tank": 37},
    "Hyundai": {"engine": 1200, "power": 83, "tank": 37},
    "Honda": {"engine": 1500, "power": 119, "tank": 40},
    "Toyota": {"engine": 2700, "power": 201, "tank": 80},
    "BMW": {"engine": 3000, "power": 258, "tank": 68},
    "Mercedes-Benz": {"engine": 3000, "power": 258, "tank": 66},
    "Audi": {"engine": 3000, "power": 245, "tank": 65},
    "Ferrari": {"engine": 3900, "power": 660, "tank": 78},
    "Rolls-Royce": {"engine": 6600, "power": 563, "tank": 82},
}

luxury_brands = ["BMW", "Mercedes-Benz", "Audi", "Ferrari", "Rolls-Royce"]

# ==================================================
# User Inputs (NO SLIDERS)
# ==================================================
st.subheader("🔧 Car Details")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", list(brand_model_map.keys()))
    model_name = st.selectbox("Model", brand_model_map[brand])

    year = st.number_input(
        "Manufacturing Year",
        min_value=1995,
        max_value=CURRENT_YEAR,
        value=2018,
        step=1
    )

    km_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        max_value=300_000,
        value=50_000,
        step=5_000
    )

with col2:
    defaults = brand_defaults[brand]

    engine_cc = st.number_input(
        "Engine Capacity (CC)",
        min_value=800,
        max_value=7000,
        value=defaults["engine"],
        step=100
    )

    max_power = st.number_input(
        "Max Power (BHP)",
        min_value=50,
        max_value=800,
        value=defaults["power"],
        step=5
    )

    fuel_tank = st.number_input(
        "Fuel Tank Capacity (Liters)",
        min_value=30,
        max_value=100,
        value=defaults["tank"],
        step=1
    )

    transmission = st.selectbox(
        "Transmission",
        ["Automatic"] if brand in luxury_brands else ["Manual", "Automatic"]
    )

    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "CNG"])

    owner = st.selectbox(
        "Owner",
        ["First Owner", "Second Owner", "Third Owner"]
    )

    color = st.selectbox(
        "Color",
        ["White", "Black", "Silver", "Grey", "Red", "Blue"]
    )

st.divider()

# ==================================================
# Unrealistic Input Guards
# ==================================================
if engine_cc < 1000 and brand in luxury_brands:
    st.error("❌ Engine capacity is unrealistic for the selected luxury brand.")
    st.stop()

# ==================================================
# Build input vector (one-hot safe)
# ==================================================
input_data = {col: 0 for col in model_columns}

input_data["Year"] = year
input_data["Kilometer"] = km_driven
input_data["engine_cc"] = engine_cc
input_data["max_power"] = max_power
input_data["Fuel_Tank_Capacity"] = fuel_tank

def set_feature(prefix, value):
    key = f"{prefix}_{value}"
    if key in input_data:
        input_data[key] = 1

set_feature("Make", brand)
set_feature("Model", model_name)
set_feature("Fuel Type", fuel)
set_feature("Transmission", transmission)
set_feature("Owner", owner)
set_feature("Color", color)

input_df = pd.DataFrame([input_data])[model_columns]

# ==================================================
# Market price caps (REALISTIC)
# ==================================================
def get_price_cap_usd(brands, engine):
    if engine <= 1200:
        return 20_000
    if 1200 < engine <= 2000:
        return 25_000
    if 2000 < engine <= 3500:
        return 60_000

    luxury_caps = {
        "BMW": 120_000,
        "Mercedes-Benz": 150_000,
        "Audi": 140_000,
        "Ferrari": 600_000,
        "Rolls-Royce": 350_000,
    }
    return luxury_caps.get(brands, 80_000)

# ==================================================
# Prediction
# ==================================================
st.subheader("💰 Prediction")

if st.button("Predict Price"):

    predictions = np.array([
        tree.predict(input_df)[0]
        for tree in model.estimators_
    ])

    mean_price = predictions.mean()
    std = predictions.std()

    lower = mean_price - std
    upper = mean_price + std

    # -----------------------------
    # Market depreciation (KEY FIX)
    # -----------------------------
    car_age = CURRENT_YEAR - year

    # Age depreciation (~6% per year)
    depreciation = max(0.35, 1 - (car_age * 0.06))

    # Mileage adjustment
    if km_driven > 150_000:
        depreciation *= 0.7
    elif km_driven > 100_000:
        depreciation *= 0.8
    elif km_driven > 60_000:
        depreciation *= 0.9

    mean_price *= depreciation
    lower *= depreciation
    upper *= depreciation

    if km_driven < car_age * 3000:
        st.warning(
            "⚠️ Mileage is unusually low for this vehicle age. "
            "Prediction may be optimistic."
        )

    cap = get_price_cap_usd(brand, engine_cc)
    mean_price = min(mean_price, cap)
    lower = max(lower, cap * 0.6)
    upper = min(upper, cap)

    st.success(f"💵 Estimated Market Price: **${mean_price:,.0f} USD**")
    st.info(f"📊 Confidence Range: **${lower:,.0f} – ${upper:,.0f} USD**")

    st.caption(
        "Prices are estimated using Indian used-car market data "
        "(converted from INR lakhs) with market calibration. "
        "Actual prices may vary."
    )

# ==================================================
# Footer
# ==================================================
st.divider()
st.caption("AI Project • Car Price Prediction • Streamlit")
