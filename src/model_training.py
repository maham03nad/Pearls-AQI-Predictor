import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import FEATURE_DATA_PATH


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "karachi_best_model.joblib"
METRICS_PATH = MODEL_DIR / "karachi_metrics.json"


def get_training_columns():
    return [
        "temperature",
        "feels_like",
        "pressure",
        "humidity",
        "wind_speed",
        "wind_degree",
        "clouds",
        "rain_1h",
        "openweather_aqi",
        "co",
        "no",
        "no2",
        "o3",
        "so2",
        "pm2_5",
        "pm10",
        "nh3",
        "hour",
        "day",
        "month",
        "day_of_week",
        "is_weekend",
        "aqi_change_rate",
        "pm25_change_rate",
        "pm10_change_rate",
        "pm25_rolling_3",
        "pm10_rolling_3",
        "aqi_rolling_3",
        "pm25_rolling_6",
        "pm10_rolling_6",
        "aqi_rolling_6",
        "aqi_lag_1",
        "pm25_lag_1",
        "pm10_lag_1",
    ]


def train_aqi_models(feature_path=FEATURE_DATA_PATH):
    print("Loading feature data...")
    df = pd.read_csv(feature_path)

    target_column = "target_aqi_next_hour"
    feature_columns = get_training_columns()

    if target_column not in df.columns:
        raise ValueError("target_aqi_next_hour column is missing. Run backfill pipeline first.")

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    models = {
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    results = {}
    best_model_name = None
    best_model = None
    best_rmse = float("inf")

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        results[model_name] = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        }

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = model_name
            best_model = model

    model_package = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_columns": feature_columns,
    }

    joblib.dump(model_package, MODEL_PATH)

    metrics = {
        "best_model": best_model_name,
        "results": results,
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print("Training completed successfully.")
    print(f"Best model: {best_model_name}")
    print(f"Model saved at: {MODEL_PATH}")
    print(f"Metrics saved at: {METRICS_PATH}")
    print(metrics)

    return metrics
