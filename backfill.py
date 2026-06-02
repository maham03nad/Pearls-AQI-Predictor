"""
Backfill Historical AQI + Weather Data
1. Fetches historical air pollution data from OpenWeather Air Pollution History API.
2. Fetches historical hourly weather from Open-Meteo Archive API.
3. Builds engineered features and AQI targets.
4. Stores rows in Hopsworks Feature Store.

Run:
    python backfill.py --days 365
"""
import os
import argparse
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import hopsworks

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# CONFIG
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")

CITY = os.getenv("CITY", "karachi")

LAT = float(os.getenv("LAT", 24.8607))
LON = float(os.getenv("LON", 67.0011))

HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqi_project_10pearls")
HOPSWORKS_PORT = int(os.getenv("HOPSWORKS_PORT", "443"))


AQI_MAP = {
    1: 25.0,
    2: 75.0,
    3: 125.0,
    4: 175.0,
    5: 300.0,
}


# FETCH AQI HISTORY
def fetch_air_pollution_history(start_dt, end_dt):
    start_unix = int(start_dt.timestamp())
    end_unix = int(end_dt.timestamp())

    url = (
        "https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={LAT}&lon={LON}&start={start_unix}&end={end_unix}&appid={OPENWEATHER_KEY}"
    )

    print("Fetching historical AQI...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    data = resp.json().get("list", [])
    print(f"Got {len(data)} records")
    return data


# BUILD POLLUTION DF
def build_pollution_rows(records):
    rows = []

    for entry in records:
        dt = datetime.fromtimestamp(entry["dt"], tz=timezone.utc).replace(tzinfo=None)

        main = entry.get("main", {})
        comp = entry.get("components", {})

        raw_aqi = int(main.get("aqi", 2))

        rows.append({
            "timestamp": dt,
            "city": CITY.capitalize(),
            "aqi": float(AQI_MAP.get(raw_aqi, 75.0)),
            "pm25": float(comp.get("pm2_5", 0)),
            "pm10": float(comp.get("pm10", 0)),
            "o3": float(comp.get("o3", 0)),
            "no2": float(comp.get("no2", 0)),
            "so2": float(comp.get("so2", 0)),
            "co": float(comp.get("co", 0)),
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# WEATHER MERGE
def merge_weather(df):
    start_date = df["timestamp"].min().strftime("%Y-%m-%d")
    end_date = df["timestamp"].max().strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m",
        "timezone": "UTC",
    }

    resp = requests.get(url, params=params, timeout=60)
    data = resp.json()["hourly"]

    weather_df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"]),
        "temp": data["temperature_2m"],
        "humidity": data["relative_humidity_2m"],
        "pressure": data["surface_pressure"],
        "wind_speed": data["wind_speed_10m"],
        "wind_deg": data["wind_direction_10m"],
    })

    df = df.sort_values("timestamp")
    weather_df = weather_df.sort_values("timestamp")

    df = pd.merge_asof(df, weather_df, on="timestamp", direction="nearest", tolerance=pd.Timedelta("1h"))

    for col in ["temp", "humidity", "pressure", "wind_speed", "wind_deg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median()).fillna(0)

    return df


# FEATURES + TARGETS
def compute_features_and_targets(df):
    df = df.sort_values("timestamp")

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int64")

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["aqi_change_rate"] = df.groupby("city")["aqi"].diff().fillna(0)
    df["aqi_rolling_6h"] = df.groupby("city")["aqi"].transform(lambda x: x.rolling(6, 1).mean())
    df["aqi_rolling_24h"] = df.groupby("city")["aqi"].transform(lambda x: x.rolling(24, 1).mean())

    df["target_aqi_3h"] = df.groupby("city")["aqi"].shift(-3)
    df["target_aqi_24h"] = df.groupby("city")["aqi"].shift(-24)
    df["target_aqi_72h"] = df.groupby("city")["aqi"].shift(-72)

    df = df.dropna(subset=[
    "target_aqi_3h",
    "target_aqi_24h",
    "target_aqi_72h"
])

    int_cols = ["hour", "day_of_week", "month", "is_weekend"]

    for col in int_cols:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0)
            .astype("int64")
        )

    return df


# STORE IN HOPSWORKS

def store_in_hopsworks(df: pd.DataFrame):
    if not HOPSWORKS_KEY:
        raise ValueError("HOPSWORKS_API_KEY is missing.")

    print(f"Storing in Hopsworks... DataFrame shape: {df.shape}")

    # BIGINT columns (must be int)
    int_cols = ["hour", "day_of_week", "month", "is_weekend"]
    for col in int_cols:
        df[col] = df[col].astype("int64")

    # DOUBLE columns (must be float)
    float_cols = [
        "humidity",
        "wind_deg",
        "temp",
        "pressure",
        "wind_speed",
        "aqi",
        "pm25",
        "pm10",
        "o3",
        "no2",
        "so2",
        "co",
        "aqi_change_rate",
        "aqi_rolling_6h",
        "aqi_rolling_24h",
        "target_aqi_3h",
        "target_aqi_24h",
        "target_aqi_72h",
    ]

    for col in float_cols:
        df[col] = df[col].astype("float64")

    # login
    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        port=HOPSWORKS_PORT,
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_KEY,
    )

    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
    )

    fg.insert(df, write_options={"wait_for_job": True})

    print(f"[OK] Inserted {len(df)} rows into feature store")

# RUN
def run(days):
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    records = fetch_air_pollution_history(start, end)
    df = build_pollution_rows(records)
    df = merge_weather(df)
    df = compute_features_and_targets(df)

    store_in_hopsworks(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()

    run(args.days)