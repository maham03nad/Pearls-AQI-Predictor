from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering import build_features


MODEL_PATH = Path("models") / "karachi_best_model.joblib"


def aqi_category(openweather_aqi):
    if openweather_aqi == 1:
        return "Good"
    if openweather_aqi == 2:
        return "Fair"
    if openweather_aqi == 3:
        return "Moderate"
    if openweather_aqi == 4:
        return "Poor"
    if openweather_aqi == 5:
        return "Very Poor"
    return "Unknown"


def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run the backfill and training pipelines first."
        )

    return joblib.load(MODEL_PATH)


def predict_next_aqi(raw_df):
    model_package = load_trained_model()

    model = model_package["model"]
    feature_columns = model_package["feature_columns"]

    feature_df = build_features(raw_df)
    X = feature_df[feature_columns]

    predictions = model.predict(X)

    result_df = feature_df[["city", "event_time", "pm2_5", "pm10", "openweather_aqi"]].copy()
    result_df["predicted_aqi_next_hour"] = predictions.round(2)
    result_df["predicted_aqi_category"] = result_df["predicted_aqi_next_hour"].round().astype(int)
    result_df["predicted_aqi_category"] = result_df["predicted_aqi_category"].clip(1, 5)
    result_df["category_name"] = result_df["predicted_aqi_category"].apply(aqi_category)

    return result_df


def predict_latest_aqi(raw_df):
    predictions = predict_next_aqi(raw_df)
    return predictions.tail(1)
