import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.sav")

pipeline = load_model()
THRESHOLD = 0.1821

# ==============================
# APP UI
# ==============================
st.title("📦 COD Failure Prediction App")
st.write("Masukkan data pesanan mentah — fitur engineering dilakukan otomatis sesuai model training.")

st.divider()

# ==============================
# USER INPUT (RAW DATA ONLY)
# ==============================
st.header("Input Data Pesanan")

waktu_pesanan = st.text_input("Waktu Pesanan Dibuat (format: YYYY-MM-DD HH:MM:SS)")
kota = st.text_input("Kota/Kabupaten (exact text)")
provinsi = st.text_input("Provinsi (exact text)")
opsi_pengiriman = st.text_input("Opsi Pengiriman (exact text)")

total_weight = st.number_input("Total Weight (gram)", 0, 5000, 200)
total_payment = st.number_input("Total Payment (Rp)", 0, 2000000, 50000)
total_qty = st.number_input("Total Quantity", 1, 50, 1)

st.divider()

# ==============================
# PREDICT
# ==============================
if st.button("🔍 Predict COD Failure"):
    
    # =============================================================
    # FEATURE ENGINEERING (seperti di training)
    # =============================================================
    dt = pd.to_datetime(waktu_pesanan, errors="coerce")

    order_month = dt.month
    order_dayofweek = dt.dayofweek
    order_hour = dt.hour
    is_weekend = int(order_dayofweek in [5, 6])

    same_city = int(kota.upper() == "KOTA TANGERANG")
    same_province = int(provinsi.upper() == "BANTEN")

    is_heavy = int(total_weight > 2000)
    is_light = int(total_weight < 250)
    high_payment = int(total_payment > 150000)
    multi_item = int(total_qty > 1)

    # =============================================================
    # DATAFRAME SESUAI TRAINING FEATURE NAMES
    # =============================================================
    input_df = pd.DataFrame([{
        "order_month": order_month,
        "order_dayofweek": order_dayofweek,
        "order_hour": order_hour,
        "is_weekend": is_weekend,

        "same_city": same_city,
        "same_province": same_province,

        "is_heavy": is_heavy,
        "is_light": is_light,
        "high_payment": high_payment,
        "multi_item": multi_item,

        "total_weight_gr": total_weight,
        "Total Pembayaran": total_payment,
        "total_qty": total_qty,

        "Opsi Pengiriman": opsi_pengiriman,
        "Kota/Kabupaten": kota,
        "Provinsi": provinsi
    }])

    # =============================================================
    # PREDICTION
    # =============================================================
    proba = pipeline.predict_proba(input_df)[0][1]
    pred = int(proba >= THRESHOLD)

    st.subheader("📊 Hasil Prediksi")
    st.write(f"**Probabilitas COD Gagal:** {proba:.4f}")
    st.write(f"**Threshold:** {THRESHOLD}")

    if pred == 1:
        st.error("⚠️ COD kemungkinan *GAGAL* ❌")
    else:
        st.success("✅ COD kemungkinan *BERHASIL*")