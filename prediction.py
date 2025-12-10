# prediction.py
import os
import json
import joblib
import pandas as pd

# Path relatif
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cod_failure_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "cod_threshold.json")

_model = None
_threshold = None

# === HARUS SAMA DENGAN DI model.py ===
numeric_features = [
    'order_month',
    'order_dayofweek',
    'order_hour',
    'is_weekend',

    'same_city',
    'same_province',
    
    'is_heavy',
    'is_light',
    'high_payment',
    'multi_item',

    'total_weight_gr',
    'Total Pembayaran',
    'total_qty'
]

categorical_features = [
    'Opsi Pengiriman',
    'Kota/Kabupaten',
    'Provinsi'
]

feature_cols = numeric_features + categorical_features


def _load_model_and_threshold():
    """Lazy load model & threshold sekali saja."""
    global _model, _threshold

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)

    if _threshold is None:
        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH, "r") as f:
                cfg = json.load(f)
                _threshold = float(cfg.get("threshold", 0.1821))
        else:
            _threshold = 0.1821

    return _model, _threshold


def predict_single(input_dict: dict):
    """
    input_dict: dict dengan key sesuai feature_cols
    """
    model, threshold = _load_model_and_threshold()

    df_input = pd.DataFrame([input_dict], columns=feature_cols)

    proba = model.predict_proba(df_input)[:, 1][0]
    label = int(proba >= threshold)

    return {
        "probability_failed": float(proba),
        "threshold": float(threshold),
        "predicted_label": label,  # 1 = risk gagal, 0 = aman
    }