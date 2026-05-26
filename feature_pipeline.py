"""
STEP 1: Feature Pipeline
- Fetches raw weather + AQI data from AQICN and OpenWeather APIs
- Engineers features (time-based + derived)
- Stores features in Hopsworks Feature Store

Notes:
- Live future target values are unknown, so target columns are stored as NaN.
- Missing pollutant values are handled using OpenWeather fallback first, then city-level defaults.
- Rolling AQI features are calculated from previous Hopsworks records when available.
"""

import os
from datetime import datetime

import hopsworks
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# CONFIG

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")

CITY = os.getenv("CITY") or "karachi"


def get_float_env(name, default):
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


LAT = get_float_env("LAT", 24.8607)
LON = get_float_env("LON", 67.0011)

HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST") or "eu-west.cloud.hopsworks.ai"
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT") or "aqi_project_10pearls"
HOPSWORKS_PORT = int(os.getenv("HOPSWORKS_PORT") or 443)

# City-level fallback values are used only if both AQICN and OpenWeather
# are missing a pollutant value. This avoids treating missing values as clean air.
CITY_DEFAULTS = {
    "pm25": 35.0,
    "pm10": 80.0,
    "o3": 50.0,
    "no2": 20.0,
    "so2": 5.0,
    "co": 300.0,
    "aqi": 125.0,
}

def safe_float(value, fallback=np.nan) -> float:
    """Safely convert API values to float."""
    try:
        if value in (None, "", "-", "NA", "N/A"):
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def safe_iaqi_value(iaqi_data: dict, key: str) -> float:
    """Return AQICN pollutant value or NaN if it is missing."""
    return safe_float(iaqi_data.get(key, {}).get("v", None), fallback=np.nan)


def fill_missing(value: float, default: float) -> float:
    """Fill missing numeric value using a city-level default."""
    return float(default) if pd.isna(value) else float(value)


def fetch_aqi_data(city: str) -> dict:
    """Fetch current AQI data from AQICN."""
    if not AQICN_TOKEN:
        raise ValueError("Missing AQICN_TOKEN")

    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    data = resp.json()
    if data.get("status") != "ok":
        raise ValueError(f"AQICN error: {data}")

    return data["data"]


def fetch_air_pollution_data(lat: float, lon: float) -> dict:
    """Fetch pollutant fallback values from OpenWeather Air Pollution API."""
    if not OPENWEATHER_KEY:
        raise ValueError("Missing OPENWEATHER_KEY")

    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}"
    )

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_weather_data(lat: float, lon: float) -> dict:
    """Fetch current weather from OpenWeatherMap."""
    if not OPENWEATHER_KEY:
        raise ValueError("Missing OPENWEATHER_KEY")

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
    )

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def engineer_features(aqi_data: dict, weather_data: dict, air_pollution_data: dict | None = None) -> dict:
    """Compute model-ready features from raw API data."""
    now = datetime.utcnow()

    iaqi = aqi_data.get("iaqi", {})

    # AQICN values are kept as NaN if missing.This avoids treating missing pollutant values as 0.The OpenWeather fallback is applied first, and only if both APIs are missing a value do we fill with city-level defaults. This ensures that we never treat missing pollutant data as 0, which could lead to misleading features.
    # pollutants to 0 because 0 may incorrectly mean "clean air".
    aqi = safe_float(aqi_data.get("aqi", None), fallback=np.nan)
    pm25 = safe_iaqi_value(iaqi, "pm25")
    pm10 = safe_iaqi_value(iaqi, "pm10")
    o3 = safe_iaqi_value(iaqi, "o3")
    no2 = safe_iaqi_value(iaqi, "no2")
    so2 = safe_iaqi_value(iaqi, "so2")
    co = safe_iaqi_value(iaqi, "co")

    # Fallback to OpenWeather Air Pollution API for any missing pollutant.
    if air_pollution_data:
        try:
            ap_item = air_pollution_data["list"][0]
            comp = ap_item.get("components", {})

            if pd.isna(aqi):
                aqi_map = {1: 25, 2: 75, 3: 125, 4: 175, 5: 300}
                aqi = float(aqi_map.get(ap_item.get("main", {}).get("aqi"), np.nan))

            if pd.isna(pm25):
                pm25 = safe_float(comp.get("pm2_5"), fallback=np.nan)
            if pd.isna(pm10):
                pm10 = safe_float(comp.get("pm10"), fallback=np.nan)
            if pd.isna(o3):
                o3 = safe_float(comp.get("o3"), fallback=np.nan)
            if pd.isna(no2):
                no2 = safe_float(comp.get("no2"), fallback=np.nan)
            if pd.isna(so2):
                so2 = safe_float(comp.get("so2"), fallback=np.nan)
            if pd.isna(co):
                co = safe_float(comp.get("co"), fallback=np.nan)
        except Exception as e:
            print(f"[!] OpenWeather pollutant fallback could not be applied: {e}")

    # Final fallback to city-level defaults only if both APIs are missing values.
    aqi = fill_missing(aqi, CITY_DEFAULTS["aqi"])
    pm25 = fill_missing(pm25, CITY_DEFAULTS["pm25"])
    pm10 = fill_missing(pm10, CITY_DEFAULTS["pm10"])
    o3 = fill_missing(o3, CITY_DEFAULTS["o3"])
    no2 = fill_missing(no2, CITY_DEFAULTS["no2"])
    so2 = fill_missing(so2, CITY_DEFAULTS["so2"])
    co = fill_missing(co, CITY_DEFAULTS["co"])

    temp = safe_float(weather_data["main"]["temp"])
    humidity = safe_float(weather_data["main"]["humidity"])
    pressure = safe_float(weather_data["main"]["pressure"])
    wind_speed = safe_float(weather_data["wind"]["speed"])
    wind_deg = safe_float(weather_data["wind"].get("deg", 0), fallback=0.0)

    hour = now.hour
    day = now.weekday()
    month = now.month
    is_weekend = int(day >= 5)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "city": CITY,
        "aqi": float(aqi),
        "pm25": float(pm25),
        "pm10": float(pm10),
        "o3": float(o3),
        "no2": float(no2),
        "so2": float(so2),
        "co": float(co),
        "temp": float(temp),
        "humidity": float(humidity),
        "pressure": float(pressure),
        "wind_speed": float(wind_speed),
        "wind_deg": float(wind_deg),
        "hour": int(hour),
        "day_of_week": int(day),
        "month": int(month),
        "is_weekend": int(is_weekend),
        "hour_sin": float(hour_sin),
        "hour_cos": float(hour_cos),
        "month_sin": float(month_sin),
        "month_cos": float(month_cos),
        #  Before insert these values are updated from Hopsworks history.
        # If history is unavailable, the current AQI remains a safe fallback.
        "aqi_change_rate": 0.0,
        "aqi_rolling_6h": float(aqi),
        "aqi_rolling_24h": float(aqi),
        # Live single-row future targets are unknown.
        # To avoids target leakage Keeping them as NaN .
        "target_aqi_3h": float("nan"),
        "target_aqi_24h": float("nan"),
        "target_aqi_72h": float("nan"),
    }

def update_live_trend_features(features: dict, feature_group):
    """
    Update live rolling features from previous Hopsworks records when available.

    Fallback:
    If reading previous records fails, keep current AQI as rolling values.
    This keeps the feature pipeline robust in scheduled runs.
    """
    try:
        history = feature_group.read()

        if history is None or history.empty:
            return features

        history = history.copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
        current_ts = pd.to_datetime(features["timestamp"], utc=True)

        history = history[
            (history["city"].astype(str).str.lower() == str(features["city"]).lower())
            & (history["timestamp"] < current_ts)
        ].sort_values("timestamp")

        if history.empty or "aqi" not in history.columns:
            return features

        recent_aqi = pd.to_numeric(history["aqi"], errors="coerce").dropna()

        if recent_aqi.empty:
            return features

        last_aqi = float(recent_aqi.iloc[-1])
        features["aqi_change_rate"] = float(features["aqi"] - last_aqi)
        features["aqi_rolling_6h"] = float(recent_aqi.tail(6).mean())
        features["aqi_rolling_24h"] = float(recent_aqi.tail(24).mean())

    except Exception as e:
        print(f"[!] Could not compute live rolling features from Hopsworks history: {e}")
        print("[!] Keeping current AQI as rolling feature fallback.")

    return features


def store_in_hopsworks(features: dict):
    """Push one row of features to the Hopsworks Feature Store."""
    if not HOPSWORKS_KEY:
        raise ValueError("Missing HOPSWORKS_API_KEY")

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
        description="Hourly AQI + weather features",
        event_time="timestamp",
    )

    features = update_live_trend_features(features, fg)

    df = pd.DataFrame([features])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fg.insert(df, write_options={"wait_for_job": False})
    print(f"[✓] Inserted 1 row at {features['timestamp']}")


def run():
    print("=== Feature Pipeline ===")
    print(f"Fetching data for {CITY}...")

    aqi_data = fetch_aqi_data(CITY)
    weather_data = fetch_weather_data(LAT, LON)

    try:
        air_pollution_data = fetch_air_pollution_data(LAT, LON)
    except Exception as e:
        print(f"[!] OpenWeather Air Pollution fallback unavailable: {e}")
        air_pollution_data = None

    features = engineer_features(aqi_data, weather_data, air_pollution_data)

    print(
        f"  AQI={features['aqi']} "
        f"PM2.5={features['pm25']} "
        f"PM10={features['pm10']} "
        f"Temp={features['temp']}°C"
    )

    store_in_hopsworks(features)
    print("=== Done ===")


if __name__ == "__main__":
    run()
