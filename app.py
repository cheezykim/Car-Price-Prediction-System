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
    layout="wide"
)

# ==================================================
# Custom CSS for modern styling (Light + Dark mode)
# ==================================================
st.markdown("""
<style>
    /* ==================== CSS Variables ==================== */
    :root {
        /* Light mode colors */
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-tertiary: #f1f5f9;
        --bg-card: #ffffff;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #64748b;
        --border-color: #e2e8f0;
        --border-light: #cbd5e1;
        --input-bg: #f8fafc;
        --input-bg-hover: #ffffff;
        --shadow-color: rgba(0, 0, 0, 0.08);
        --info-bg: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        --info-border: #bfdbfe;
        --info-text: #1e40af;
    }

    /* Dark mode colors */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #1e293b;
            --bg-secondary: #0f172a;
            --bg-tertiary: #334155;
            --bg-card: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --border-light: #475569;
            --input-bg: #1e293b;
            --input-bg-hover: #334155;
            --shadow-color: rgba(0, 0, 0, 0.3);
            --info-bg: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
            --info-border: #1e3a5f;
            --info-text: #7dd3fc;
        }
    }

    /* ==================== Hide Streamlit Elements ==================== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ==================== Main Container ==================== */
    .block-container {
        padding: 1rem 2rem 2rem 2rem;
        max-width: 1400px;
    }

    /* ==================== Hero Section ==================== */
    .hero-section {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0ea5e9 100%);
        padding: 2rem 2.5rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.4);
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.5;
    }
    .hero-content {
        position: relative;
        z-index: 1;
    }
    .hero-section h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .hero-section p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* ==================== Section Titles ==================== */
    .section-title {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--border-color);
    }
    .section-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    .section-icon.blue { background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); }
    .section-icon.green { background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); }
    .section-icon.orange { background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%); }
    .section-icon.purple { background: linear-gradient(135deg, #e9d5ff 0%, #d8b4fe 100%); }
    .section-label {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }

    /* ==================== Panel Header ==================== */
    .panel-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.25rem;
    }
    .panel-icon {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
    }
    .panel-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    .panel-subtitle {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* ==================== Car Summary Card ==================== */
    .car-summary {
        background: var(--bg-tertiary);
        padding: 1.25rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        border: 1px solid var(--border-light);
    }
    .car-summary h4 {
        margin: 0;
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 700;
    }
    .car-summary p {
        margin: 0.5rem 0 0 0;
        color: var(--text-secondary);
        font-size: 0.85rem;
    }
    .car-specs {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.75rem;
        flex-wrap: wrap;
    }
    .spec-tag {
        background: var(--bg-card);
        padding: 0.3rem 0.7rem;
        border-radius: 6px;
        font-size: 0.75rem;
        color: var(--text-secondary);
        border: 1px solid var(--border-color);
    }

    /* ==================== Price Display ==================== */
    .price-display {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        padding: 1.75rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 15px 40px rgba(15, 23, 42, 0.3);
        position: relative;
        overflow: hidden;
    }
    .price-display::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(14,165,233,0.2) 0%, transparent 70%);
    }
    .price-content {
        position: relative;
        z-index: 1;
    }
    .price-label {
        font-size: 0.8rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 500;
    }
    .price-value {
        font-size: 2.75rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.1;
        letter-spacing: -1px;
    }
    .price-currency {
        font-size: 0.85rem;
        opacity: 0.7;
        margin-top: 0.25rem;
        font-weight: 500;
    }

    /* ==================== Estimate Cards ==================== */
    .estimate-row {
        display: flex;
        gap: 0.75rem;
        margin-top: 1rem;
    }
    .estimate-card {
        flex: 1;
        background: var(--bg-card);
        padding: 1rem;
        border-radius: 14px;
        text-align: center;
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
    }
    .estimate-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px var(--shadow-color);
    }
    .estimate-card.low {
        border-bottom: 3px solid #f59e0b;
    }
    .estimate-card.high {
        border-bottom: 3px solid #10b981;
    }
    .estimate-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }
    .estimate-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    /* ==================== Placeholder Card ==================== */
    .placeholder-card {
        background: var(--bg-tertiary);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        border: 2px dashed var(--border-light);
        margin: 1rem;
    }
    .placeholder-icon {
        font-size: 3rem;
        margin-bottom: 0.75rem;
        filter: grayscale(0.3);
    }
    .placeholder-text {
        color: var(--text-muted);
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .placeholder-text strong {
        color: #0ea5e9;
    }

    /* ==================== Button Styling ==================== */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        border: none;
        padding: 0.9rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 14px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.35);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(14, 165, 233, 0.45);
        background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
    }
    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* ==================== Input Styling (Light Mode) ==================== */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        border-radius: 12px !important;
        border-color: var(--border-color) !important;
        background: var(--input-bg) !important;
        color: var(--text-primary) !important;
        transition: all 0.2s ease;
    }
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div:hover {
        border-color: var(--border-light) !important;
        background: var(--input-bg-hover) !important;
    }
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div:focus-within {
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.15) !important;
        background: var(--input-bg-hover) !important;
    }

    /* Selectbox dropdown text */
    .stSelectbox [data-baseweb="select"] span {
        color: var(--text-primary) !important;
    }

    /* Labels */
    .stSelectbox label, .stNumberInput label {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        margin-bottom: 0.3rem !important;
    }

    /* ==================== Streamlit Container Borders ==================== */
    [data-testid="stVerticalBlock"] > div:has(> div.stMarkdown) {
        border-color: var(--border-color) !important;
    }

    /* ==================== Divider ==================== */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 1.25rem 0;
    }

    /* ==================== Footer ==================== */
    .footer {
        text-align: center;
        padding: 2.5rem 0 5rem 0;
        color: var(--text-muted);
        font-size: 0.85rem;
    }
    .footer-content {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }

    /* ==================== Info Tip ==================== */
    .info-tip {
        background: var(--info-bg);
        border: 1px solid var(--info-border);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        font-size: 0.85rem;
        color: var(--info-text);
        margin: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
    }
    .info-tip-icon {
        flex-shrink: 0;
    }

    /* ==================== Dark Mode Specific Overrides ==================== */
    @media (prefers-color-scheme: dark) {
        /* Streamlit native elements */
        .stApp {
            background-color: var(--bg-secondary);
        }

        /* Container borders */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: var(--bg-card) !important;
            border-color: var(--border-color) !important;
        }

        /* Selectbox dropdown menu */
        [data-baseweb="popover"] {
            background-color: var(--bg-card) !important;
        }
        [data-baseweb="popover"] li {
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
        }
        [data-baseweb="popover"] li:hover {
            background-color: var(--bg-tertiary) !important;
        }

        /* Number input */
        .stNumberInput input {
            color: var(--text-primary) !important;
        }

        /* Selectbox SVG arrow */
        .stSelectbox svg {
            fill: var(--text-muted) !important;
        }

        /* Warning box */
        .stAlert {
            background-color: rgba(251, 191, 36, 0.1) !important;
            border-color: #f59e0b !important;
        }

        /* Section icons - slightly muted in dark mode */
        .section-icon.blue { background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 100%); }
        .section-icon.green { background: linear-gradient(135deg, #064e3b 0%, #065f46 100%); }
        .section-icon.orange { background: linear-gradient(135deg, #78350f 0%, #92400e 100%); }
        .section-icon.purple { background: linear-gradient(135deg, #4c1d95 0%, #5b21b6 100%); }
    }

    /* ==================== Responsive Adjustments ==================== */
    @media (max-width: 768px) {
        .hero-section h1 { font-size: 1.8rem; }
        .price-value { font-size: 2.2rem; }
        .block-container { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

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
# Hero Section
# ==================================================
st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <div class="hero-badge">✨ AI-Powered Valuation</div>
        <h1>🚗 Car Price Prediction</h1>
        <p>Get instant, accurate market valuations powered by machine learning</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# Main Content Layout
# ==================================================
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    # ==================================================
    # Basic Information Card
    # ==================================================
    with st.container(border=True):
        st.markdown("""
        <div class="section-title">
            <div class="section-icon blue">🚙</div>
            <h3 class="section-label">Basic Information</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            brand = st.selectbox("Brand", list(brand_model_map.keys()))
        with col2:
            model_name = st.selectbox("Model", brand_model_map[brand])
        with col3:
            year = st.number_input(
                "Manufacturing Year",
                min_value=1995,
                max_value=CURRENT_YEAR,
                value=2018,
                step=1
            )

    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

    # ==================================================
    # Technical Specifications Card
    # ==================================================
    with st.container(border=True):
        st.markdown("""
        <div class="section-title">
            <div class="section-icon green">⚙️</div>
            <h3 class="section-label">Technical Specifications</h3>
        </div>
        """, unsafe_allow_html=True)

        defaults = brand_defaults[brand]
        col1, col2, col3 = st.columns(3)

        with col1:
            engine_cc = st.number_input(
                "Engine Capacity (CC)",
                min_value=800,
                max_value=7000,
                value=defaults["engine"],
                step=100
            )
        with col2:
            max_power = st.number_input(
                "Max Power (BHP)",
                min_value=50,
                max_value=800,
                value=defaults["power"],
                step=5
            )
        with col3:
            fuel_tank = st.number_input(
                "Fuel Tank Capacity (Liters)",
                min_value=30,
                max_value=100,
                value=defaults["tank"],
                step=1
            )

    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

    # ==================================================
    # Usage & Condition Card
    # ==================================================
    with st.container(border=True):
        st.markdown("""
        <div class="section-title">
            <div class="section-icon orange">📊</div>
            <h3 class="section-label">Usage & Condition</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            km_driven = st.number_input(
                "Kilometers Driven",
                min_value=0,
                max_value=300_000,
                value=50_000,
                step=5_000
            )
        with col2:
            transmission = st.selectbox(
                "Transmission",
                ["Automatic"] if brand in luxury_brands else ["Manual", "Automatic"]
            )
        with col3:
            fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "CNG"])

    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

    # ==================================================
    # Additional Details Card
    # ==================================================
    with st.container(border=True):
        st.markdown("""
        <div class="section-title">
            <div class="section-icon purple">✨</div>
            <h3 class="section-label">Additional Details</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            owner = st.selectbox(
                "Owner",
                ["First Owner", "Second Owner", "Third Owner"]
            )
        with col2:
            color = st.selectbox(
                "Color",
                ["White", "Black", "Silver", "Grey", "Red", "Blue"]
            )

# ==================================================
# Unrealistic Input Guards
# ==================================================
if engine_cc < 1000 and brand in luxury_brands:
    st.error("Engine capacity is unrealistic for the selected luxury brand.")
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
# Prediction Panel (Right Column)
# ==================================================
with right_col:
    with st.container(border=True):
        st.markdown("""
        <div class="panel-header">
            <div class="panel-icon">💰</div>
            <div>
                <h3 class="panel-title">Price Prediction</h3>
                <p class="panel-subtitle">AI-powered market valuation</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Car Summary Card
        st.markdown(f"""
        <div class="car-summary">
            <h4>{year} {brand} {model_name}</h4>
            <p>{km_driven:,} km  •  {fuel}  •  {transmission}</p>
            <div class="car-specs">
                <span class="spec-tag">{engine_cc} CC</span>
                <span class="spec-tag">{max_power} BHP</span>
                <span class="spec-tag">{owner}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)

        if predict_button:
            with st.spinner("Analyzing market data..."):
                predictions = np.array([
                    tree.predict(input_df)[0]
                    for tree in model.estimators_
                ])

                mean_price = predictions.mean()
                std = predictions.std()

                lower = mean_price - std
                upper = mean_price + std

                # Market depreciation
                car_age = CURRENT_YEAR - year
                depreciation = max(0.35, 1 - (car_age * 0.06))

                if km_driven > 150_000:
                    depreciation *= 0.7
                elif km_driven > 100_000:
                    depreciation *= 0.8
                elif km_driven > 60_000:
                    depreciation *= 0.9

                mean_price *= depreciation
                lower *= depreciation
                upper *= depreciation

                cap = get_price_cap_usd(brand, engine_cc)
                mean_price = min(mean_price, cap)
                lower = max(lower, cap * 0.6)
                upper = min(upper, cap)

            # Main Price Display
            st.markdown(f"""
            <div class="price-display">
                <div class="price-content">
                    <div class="price-label">Estimated Market Price</div>
                    <div class="price-value">${mean_price:,.0f}</div>
                    <div class="price-currency">United States Dollar</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Estimate Range
            st.markdown(f"""
            <div class="estimate-row">
                <div class="estimate-card low">
                    <div class="estimate-label">Low Estimate</div>
                    <div class="estimate-value">${lower:,.0f}</div>
                </div>
                <div class="estimate-card high">
                    <div class="estimate-label">High Estimate</div>
                    <div class="estimate-value">${upper:,.0f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Warning for low mileage
            if km_driven < car_age * 3000:
                st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
                st.warning(
                    "⚠️ Mileage is unusually low for this vehicle age. "
                    "Prediction may be optimistic."
                )

            st.markdown("""
            <div class="info-tip">
                <span class="info-tip-icon">💡</span>
                <span>Prices are estimated using Indian used-car market data with market calibration. Actual prices may vary based on condition, location, and market demand.</span>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Placeholder when no prediction made yet
            st.markdown("""
            <div class="placeholder-card">
                <div class="placeholder-icon">🚗</div>
                <div class="placeholder-text">
                    Enter your car details<br>
                    and click <strong>Predict Price</strong><br>
                    to get an instant valuation
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==================================================
# Footer
# ==================================================
st.markdown("""
<div class="footer">
    <div class="footer-content">
        <span>Built with</span>
        <span>❤️</span>
        <span>using Streamlit & Machine Learning</span>
    </div>
</div>
""", unsafe_allow_html=True)
