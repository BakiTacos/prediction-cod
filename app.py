import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# LOAD PIPELINE + MODEL
# ==============================
@st.cache_resource
def load_model():
    pipeline = joblib.load("xgboost_model.sav")
    return pipeline

pipeline = load_model()

# Custom threshold (harus sama seperti waktu training)
THRESHOLD = 0.1821

# ==============================
# STREAMLIT UI
# ==============================
st.title("📦 COD Failure Prediction App")
st.write("Prediksi apakah pesanan COD akan **gagal** berdasarkan fitur e-commerce Indonesia.")

st.divider()

# ==============================
# INPUT FORM
# ==============================
st.header("Input Order Features")

col1, col2 = st.columns(2)

with col1:
    order_month = st.number_input("Order Month (1–12)", 1, 12, 1)
    order_dayofweek = st.number_input("Day of Week (0=Mon ... 6=Sun)", 0, 6, 0)
    order_hour = st.number_input("Order Hour (0–23)", 0, 23, 12)
    is_weekend = st.selectbox("Is Weekend?", [0, 1])

    same_city = st.selectbox("Same City?", [0, 1])
    same_province = st.selectbox("Same Province?", [0, 1])

    is_heavy = st.selectbox("Is Heavy (>2000gr)?", [0, 1])
    is_light = st.selectbox("Is Light (<250gr)?", [0, 1])

with col2:
    high_payment = st.selectbox("High Payment (>150k)?", [0, 1])
    multi_item = st.selectbox("Multiple Items?", [0, 1])

    total_weight_gr = st.number_input("Total Weight (gram)", 0, 5000, 200)
    total_payment = st.number_input("Total Payment (Rp)", 0, 2000000, 50000)
    total_qty = st.number_input("Total Quantity", 1, 20, 1)

    opsi_pengiriman = st.text_input("Opsi Pengiriman (exact text)")
    kota = st.text_input("Kota/Kabupaten (exact text)")
    provinsi = st.text_input("Provinsi (exact text)")

st.divider()

# ==============================
# PREDICT BUTTON
# ==============================
if st.button("🔍 Predict COD Failure"):
    
    input_data = pd.DataFrame([{
        'order_month': order_month,
        'order_dayofweek': order_dayofweek,
        'order_hour': order_hour,
        'is_weekend': is_weekend,

        'same_city': same_city,
        'same_province': same_province,

        'is_heavy': is_heavy,
        'is_light': is_light,
        'high_payment': high_payment,
        'multi_item': multi_item,

        'total_weight_gr': total_weight_gr,
        'Total Pembayaran': total_payment,
        'total_qty': total_qty,

        'Opsi Pengiriman': opsi_pengiriman,
        'Kota/Kabupaten': kota,
        'Provinsi': provinsi
    }])

    # Predict probability
    proba = pipeline.predict_proba(input_data)[0][1]

    # Apply threshold
    label = int(proba >= THRESHOLD)

    st.subheader("📊 Prediction Result")
    st.write(f"**Probability COD Failure:** {proba:.4f}")
    st.write(f"**Threshold:** {THRESHOLD}")

    if label == 1:
        st.error("⚠️ COD kemungkinan **GAGAL** ❌")
    else:
        st.success("✅ COD kemungkinan **BERHASIL**")