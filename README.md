# 🚗 AI Car Price Prediction System

This project is an **AI-powered used car price prediction web application** built with **Machine Learning and Streamlit**.

The model predicts **realistic used-car prices in USD**, calibrated to real-world market behavior using:
- Engine-based segmentation
- Brand-based price caps
- Age & mileage depreciation

---

## 📌 Features

- Random Forest Regressor model
- Prices converted from **Indian market data (INR lakhs → USD)**
- Market calibration layer (prevents unrealistic prices)
- Confidence range estimation
- Smart input defaults by brand
- Clean, slider-free Streamlit UI

---

## 🧠 Model Overview

- **Algorithm:** Random Forest Regressor  
- **Target:** Car Price (USD)  
- **Training Data:** Indian used-car dataset  
- **Evaluation:** R² ≈ 0.93  

The ML model learns pricing patterns, while a **post-prediction calibration layer** ensures real-world realism.

---

## 🗂️ Project Structure

```
car-price-prediction/
│
├── app.py                  # Streamlit web app
├── car_price_model.pkl     # Trained ML model
├── model_columns.pkl       # Feature columns
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## ▶️ How to Run the App

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Streamlit
```bash
streamlit run app.py
```

### 3️⃣ Open browser
```
http://localhost:8501
```

---

## 🧪 Example Test Case

**Honda City (2017)**
- Engine: 1500 cc
- Mileage: 60,000 km
- Transmission: Automatic

➡️ Predicted price: **$10,000 – $14,000 USD**

---

## ⚠️ Disclaimer

Prices are **estimates only** based on historical data and market calibration.
Actual market prices may vary depending on location and condition.

---

## 🎓 Academic Note

This project was developed as part of an **AI / Machine Learning assignment** to demonstrate:
- End-to-end ML pipeline
- Model evaluation & calibration
- Real-world deployment with Streamlit

---

## 👤 Author

Student AI Project  
Car Price Prediction using Machine Learning & Streamlit
