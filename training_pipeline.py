"""
AQI TRAINING PIPELINE 
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import hopsworks

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from dotenv import load_dotenv
load_dotenv()


# ================= CONFIG =================
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST") or "eu-west.cloud.hopsworks.ai"
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT") or "aqi_project"

if not HOPSWORKS_KEY:
    raise ValueError("Missing HOPSWORKS_API_KEY")

TARGET_COL = "target_aqi_72h"

BASE_FEATURES = [
    "pm25","pm10","o3","no2","so2","co",
    "temp","humidity","pressure","wind_speed","wind_deg",
    "hour","day_of_week","month","is_weekend",
    "hour_sin","hour_cos","month_sin","month_cos",
]

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ================= LOAD =================
def load_data():
    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_KEY,
    )

    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)

    df = fg.read()
    print(f"[✓] Loaded {len(df)} rows")
    return df


# ================= FEATURE ENGINEERING =================
def feature_engineering(df):

    df = df.sort_values("timestamp").reset_index(drop=True)

    # -------- LAG FEATURES --------
    for lag in [1,3,6,12]:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

    # -------- DIFFERENCE --------
    df["aqi_diff_1"] = df["aqi"].diff(1)
    df["aqi_diff_6"] = df["aqi"].diff(6)

    # -------- ROLLING FEATURES (NO LEAKAGE) --------
    df["aqi_roll_mean_6"] = df["aqi"].rolling(6).mean().shift(1)
    df["aqi_roll_mean_24"] = df["aqi"].rolling(24).mean().shift(1)

    # -------- MOMENTUM --------
    df["pm25_momentum"] = df["pm25"] - df["pm25"].shift(6)
    df["pm10_momentum"] = df["pm10"] - df["pm10"].shift(6)

    # -------- WEATHER SHIFTED FEATURES --------
    df["wind_speed_lag_3"] = df["wind_speed"].shift(3)
    df["humidity_lag_3"] = df["humidity"].shift(3)
    df["pressure_lag_6"] = df["pressure"].shift(6)

    # -------- SMOOTH WEATHER --------
    df["pressure_roll_12"] = df["pressure"].rolling(12).mean()
    df["wind_roll_6"] = df["wind_speed"].rolling(6).mean()

    # -------- INSTABILITY INDEX --------
    df["instability_index"] = df["wind_speed"] * df["pressure"].diff().abs()

    # -------- CLEAN --------
    df = df[(df["aqi"] >= 0) & (df["aqi"] <= 500)]
    df = df.dropna()

    print("[✓] Feature engineering done:", len(df))
    return df

# ================= FEATURES =================
def get_features(df):

    lag_features = [
        "aqi_lag_1","aqi_lag_3","aqi_lag_6","aqi_lag_12",
        "aqi_diff_1","aqi_diff_6",
        "aqi_roll_mean_6","aqi_roll_mean_24",
        "pm25_momentum","pm10_momentum",
        "wind_speed_lag_3","humidity_lag_3","pressure_lag_6",
        "pressure_roll_12","wind_roll_6",
        "instability_index"
    ]

    return BASE_FEATURES + lag_features

# ================= SPLIT =================
def split(df, FEATURES):

    split_idx = int(len(df) * 0.8)

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[FEATURES].values
    y_train = train[TARGET_COL].values

    X_test = test[FEATURES].values
    y_test = test[TARGET_COL].values

    return X_train, X_test, y_train, y_test

# ================= EVAL =================
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"{name}: RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.4f}")

    return {"model": name, "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

# ================= MODELS =================
def train_models(X_train, X_test, y_train, y_test):

    models = {
        "RF": RandomForestRegressor(n_estimators=300, random_state=42),
        "GB": GradientBoostingRegressor(),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=5))
        ])
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results[name] = evaluate(name, y_test, preds)

        joblib.dump(model, f"{MODELS_DIR}/{name}.pkl")

    return results

# ================= LSTM =================
def train_lstm(X, y, seq_len=24):

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_seq, y_seq = [], []

    for i in range(seq_len, len(Xs)):
        X_seq.append(Xs[i-seq_len:i])
        y_seq.append(y[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
    )

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, X.shape[1])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=32,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    preds = model.predict(X_test).flatten()

    metrics = evaluate("LSTM", y_test, preds)

    model.save(f"{MODELS_DIR}/lstm.keras")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")

    return metrics


# ============ MAIN =================
def run():

    print("\n🚀 AQI PIPELINE STARTED\n")

    df = load_data()

    df = feature_engineering(df)

    FEATURES = get_features(df)

    X_train, X_test, y_train, y_test = split(df, FEATURES)

    print("\n--- MODELS ---")
    results = train_models(X_train, X_test, y_train, y_test)

    print("\n--- LSTM ---")
    results["LSTM"] = train_lstm(df[FEATURES].values, df[TARGET_COL].values)

    print("\n--- FINAL RESULTS ---")
    for k, v in sorted(results.items(), key=lambda x: x[1]["rmse"]):
        print(k, v)

    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✔ PIPELINE COMPLETE")


if __name__ == "__main__":
    run()