import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_model():
    model = joblib.load("xgboost_model.sav")
    return model

model_xgb = load_model() 

st.title("Prediksi Risiko COD Gagal Kirim")

st.write("Model ini memprediksi probabilitas pesanan COD mengalami "
         "**Pengiriman Gagal / Paket Hilang** berdasarkan data pesanan.")

with st.form("cod_form"):
    col1, col2 = st.columns(2)

    with col1:
        order_month = st.number_input("Bulan Pesanan (1-12)", 1, 12, 12)
        
        # Bisa dibuat lebih user-friendly (teks hari), tapi untuk match model: 0–6
        order_dayofweek = st.number_input("Hari (0=Senin ... 6=Minggu)", 0, 6, 5)
        
        order_hour = st.number_input("Jam Pesanan (0-23)", 0, 23, 20)

        is_weekend = st.selectbox("Weekend?", ["Tidak", "Ya"])
        same_city = st.selectbox("Apakah Kota/Kabupaten = KOTA TANGERANG?", ["Tidak", "Ya"])
        same_province = st.selectbox("Apakah Provinsi = BANTEN?", ["Tidak", "Ya"])

    with col2:
        total_weight_gr = st.number_input("Total berat (gram)", 1, 50000, 500)
        total_payment = st.number_input("Total pembayaran (Rp)", 1, 5000000, 50000)
        total_qty = st.number_input("Total qty", 1, 100, 1)

        opsi_pengiriman = st.text_input("Opsi Pengiriman (sesuai dataset, contoh: Reguler SPX)", "Reguler")
        kota = st.text_input("Kota/Kabupaten", "KOTA JAKARTA BARAT")
        provinsi = st.text_input("Provinsi", "DKI JAKARTA")

    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_dict = {
        # numeric_features
        "order_month": int(order_month),
        "order_dayofweek": int(order_dayofweek),
        "order_hour": int(order_hour),
        "is_weekend": 1 if is_weekend == "Ya" else 0,

        "same_city": 1 if same_city == "Ya" else 0,
        "same_province": 1 if same_province == "Ya" else 0,

        "is_heavy": 1 if total_weight_gr > 2000 else 0,
        "is_light": 1 if total_weight_gr < 250 else 0,
        "high_payment": 1 if total_payment > 150000 else 0,
        "multi_item": 1 if total_qty > 1 else 0,

        "total_weight_gr": float(total_weight_gr),
        "Total Pembayaran": float(total_payment),
        "total_qty": int(total_qty),

        # categorical_features
        "Opsi Pengiriman": opsi_pengiriman,
        "Kota/Kabupaten": kota,
        "Provinsi": provinsi,
    }

    def predict_single(feature_row: pd.DataFrame):
    # model_xgb adalah Pipeline(preprocess + model)
        proba = model_xgb.predict_proba(feature_row)[:, 1][0]
        pred = int(proba >= 0.3)  # threshold yang sama dengan script
        return pred, proba

    result = predict_single(input_dict)

    proba = result["probability_failed"]
    label = result["predicted_label"]
    threshold = result["threshold"]

    st.subheader("Hasil Prediksi")
    st.write(f"Threshold model: **{threshold:.4f}**")
    st.write(f"Probabilitas COD gagal kirim: **{proba:.2%}**")

    if label == 1:
        st.error("⚠ Pesanan ini berisiko **TINGGI** mengalami Pengiriman Gagal / Paket Hilang.")
    else:
        st.success("✅ Pesanan ini **relatif aman** (risiko rendah).")