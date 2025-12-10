import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter

import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "all_months_clean.csv"

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "bakitacos/indonesia-e-commerce-sales-and-shipping-20232025",
    file_path,
    pandas_kwargs={
        "sep": ";",          # ganti pemisah
        "engine": "python",  # parser yang lebih fleksibel
        "on_bad_lines": "error",  # biar kelihatan jelas kalau masih ada baris bermasalah
    },
)

df = df[df['Metode Pembayaran'] == 'COD (Bayar di Tempat)'].copy()
df = df.drop(columns=['source_file'])

df['Alasan Pembatalan'] = df['Alasan Pembatalan'].fillna('-')

def classify_cod_failure(reason):
    reason = reason.lower()

    # logistic failure (TARGET = 1)
    if "paket hilang" in reason:
        return 1
    if "pengiriman gagal" in reason:
        return 1

    # otherwise -> TARGET = 0
    return 0

df['cod_failed'] = df['Alasan Pembatalan'].apply(classify_cod_failure)

df = df.dropna(subset=['Waktu Pesanan Dibuat'])

df['order_time'] = pd.to_datetime(df['Waktu Pesanan Dibuat'], errors='coerce')

df['order_year'] = df['order_time'].dt.year
df['order_month'] = df['order_time'].dt.month
df['order_day'] = df['order_time'].dt.day
df['order_dayofweek'] = df['order_time'].dt.dayofweek   # 0=Senin, 6=Minggu
df['order_hour'] = df['order_time'].dt.hour
df['is_weekend'] = df['order_dayofweek'].isin([5, 6]).astype(int)

df['same_city'] = (df['Kota/Kabupaten'] == 'KOTA TANGERANG')

df['same_province'] = (df['Provinsi'] == 'BANTEN')

df['is_heavy'] = (df['total_weight_gr'] > 2000).astype(int)  # > 2 kg contoh saja
df['is_light'] = (df['total_weight_gr'] < 250).astype(int)    # sangat rawan hilang
df['high_payment'] = (df['Total Pembayaran'] > 150000).astype(int)
df['multi_item'] = (df['total_qty'] > 1).astype(int)

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

X = df[feature_cols]
y = df['cod_failed']
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ]
)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=36,  # 9700/264 contoh
    eval_metric='logloss'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

pipeline.fit(X_train, y_train)

pred_proba = pipeline.predict_proba(X_test)[:, 1]
pred_class = (pred_proba >= 0.1821).astype(int)

import joblib
import json

# Simpan pipeline
joblib.dump(pipeline, "cod_failure_model.pkl")

# Simpan threshold (0.1821) ke file terpisah
threshold = 0.1821
with open("cod_threshold.json", "w") as f:
    json.dump({"threshold": threshold}, f)

print("Model & threshold saved.")