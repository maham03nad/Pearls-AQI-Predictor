"""
STEP 1: Feature Pipeline
- Fetches raw weather + AQI data from AQICN API
- Uses OpenWeather Air Pollution as fallback for missing pollutant values
- Engineers time-based, cyclical, rolling, and target features
- Stores features in Hopsworks Feature Store

Run locally:
    python feature_pipeline.py
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import hopsworks
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

#  CONFIG  

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")

CITY = os.getenv("CITY") or "karachi"

def get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default

LAT = get_float_env("LAT", 24.8607)
LON = get_float_env("LON", 67.0011)

HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST") or "eu-west.cloud.hopsworks.ai"
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT") or "aqi_project_10pearls"
HOPSWORKS_PORT = int(os.getenv("HOPSWORKS_PORT") or 443)

LAST_KNOWN_PATH = Path("last_known_values.json")

# City-level defaults are used only when both APIs and last-known cache are missing.
DEFAULT_VALUES = {
    "aqi": 125.0,
    "pm25": 35.0,
    "pm10": 80.0,
    "o3": 50.0,
    "no2": 20.0,
    "so2": 5.0,
    "co": 300.0,
}

# OpenWeather Air Pollution API returns AQI as an ordinal 1–5 category.
# This mapping is an approximate fallback only, not a replacement for AQICN AQI.
OPENWEATHER_AQI_MAP = {
    1: 25.0,
    2: 75.0,
    3: 125.0,
    4: 175.0,
    5: 300.0,
}
#  SAFE VALUE HELPERS  

def safe_float(value: Any, fallback: Optional[float] = None) -> Optional[float]:
    """Convert API value to float safely."""
    try:
        if value in (None, "", "-", "NaN"):
            return fallback
        result = float(value)
        if np.isnan(result):
            return fallback
        return result
    except (TypeError, ValueError):
        return fallback


def load_last_known_values() -> dict:
    """Load last successful API values from local cache."""
    if not LAST_KNOWN_PATH.exists():
        return {}

    try:
        with open(LAST_KNOWN_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_last_known_values(values: dict) -> None:
    """Save latest successful values so future runs can use them if APIs miss fields."""
    try:
        with open(LAST_KNOWN_PATH, "w", encoding="utf-8") as f:
            json.dump(values, f, indent=2)
    except Exception as e:
        print(f"[!] Could not save {LAST_KNOWN_PATH}: {e}")


def get_aqicn_pollutant(iaqi: dict, key: str) -> Optional[float]:
    """Read pollutant value from AQICN iaqi block."""
    return safe_float(iaqi.get(key, {}).get("v"))


def get_openweather_pollutant(air_pollution_data: Optional[dict], key: str) -> Optional[float]:
    """Read pollutant value from OpenWeather Air Pollution API."""
    if not air_pollution_data:
        return None

    ow_key_map = {
        "pm25": "pm2_5",
        "pm10": "pm10",
        "o3": "o3",
        "no2": "no2",
        "so2": "so2",
        "co": "co",
    }

    try:
        comp = air_pollution_data["list"][0]["components"]
        return safe_float(comp.get(ow_key_map[key]))
    except Exception:
        return None

def resolve_value(name: str, primary: Optional[float], fallback: Optional[float], last_known: dict) -> float:
    """
    Choose best available value:
    1. Primary AQICN value
    2. OpenWeather fallback value
    3. Last-known cached value
    4. City-level default value
    """
    if primary is not None:
        return float(primary)

    if fallback is not None:
        return float(fallback)

    cached = safe_float(last_known.get(name))
    if cached is not None:
        return float(cached)

    return float(DEFAULT_VALUES.get(name, 0.0))

#  API FETCHING  

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


def fetch_air_pollution_data(lat: float, lon: float) -> Optional[dict]:
    """Fetch pollutant fallback data from OpenWeather Air Pollution API."""
    if not OPENWEATHER_KEY:
        return None

    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[!] OpenWeather Air Pollution fallback failed: {e}")
        return None

#  FEATURE ENGINEERING 

def engineer_features(
    aqi_data: dict,
    weather_data: dict,
    air_pollution_data: Optional[dict] = None,
) -> dict:
    """Compute model-ready features from raw API data."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    last_known = load_last_known_values()

    iaqi = aqi_data.get("iaqi", {})

    # Primary AQI from AQICN
    aqi_primary = safe_float(aqi_data.get("aqi"))

    # Fallback AQI from OpenWeather ordinal scale
    aqi_fallback = None
    if air_pollution_data:
        try:
            ow_aqi_class = int(air_pollution_data["list"][0]["main"]["aqi"])
            aqi_fallback = OPENWEATHER_AQI_MAP.get(ow_aqi_class)
        except Exception:
            aqi_fallback = None

    aqi = resolve_value("aqi", aqi_primary, aqi_fallback, last_known)

    # Pollutants: AQICN → OpenWeather → last-known → default
    pm25 = resolve_value(
        "pm25",
        get_aqicn_pollutant(iaqi, "pm25"),
        get_openweather_pollutant(air_pollution_data, "pm25"),
        last_known,
    )
    pm10 = resolve_value(
        "pm10",
        get_aqicn_pollutant(iaqi, "pm10"),
        get_openweather_pollutant(air_pollution_data, "pm10"),
        last_known,
    )
    o3 = resolve_value(
        "o3",
        get_aqicn_pollutant(iaqi, "o3"),
        get_openweather_pollutant(air_pollution_data, "o3"),
        last_known,
    )
    no2 = resolve_value(
        "no2",
        get_aqicn_pollutant(iaqi, "no2"),
        get_openweather_pollutant(air_pollution_data, "no2"),
        last_known,
    )
    so2 = resolve_value(
        "so2",
        get_aqicn_pollutant(iaqi, "so2"),
        get_openweather_pollutant(air_pollution_data, "so2"),
        last_known,
    )
    co = resolve_value(
        "co",
        get_aqicn_pollutant(iaqi, "co"),
        get_openweather_pollutant(air_pollution_data, "co"),
        last_known,
    )

    temp = safe_float(weather_data.get("main", {}).get("temp"), 0.0)
    humidity = safe_float(weather_data.get("main", {}).get("humidity"), 0.0)
    pressure = safe_float(weather_data.get("main", {}).get("pressure"), 0.0)
    wind_speed = safe_float(weather_data.get("wind", {}).get("speed"), 0.0)
    wind_deg = safe_float(weather_data.get("wind", {}).get("deg"), 0.0)

    hour = now.hour
    day = now.weekday()
    month = now.month
    is_weekend = int(day >= 5)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    latest_values = {
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
        "updated_at": now.isoformat(),
    }
    save_last_known_values(latest_values)

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

        # These are updated using recent Feature Store history before insert.
        # Safe fallback values are used if history is unavailable.
        "aqi_change_rate": 0.0,
        "aqi_rolling_6h": float(aqi),
        "aqi_rolling_24h": float(aqi),

        # Future AQI values are unknown for a live row.
        # Keep targets null to avoid target leakage.
        "target_aqi_3h": None,
        "target_aqi_24h": None,
        "target_aqi_72h": None,
    }

#  LIVE TREND FEATURES 

def update_live_trend_features(features: dict, fg) -> dict:
    """
    Update aqi_change_rate, aqi_rolling_6h, and aqi_rolling_24h using recent history.

    If Hopsworks history cannot be read, the function keeps safe fallback values
    based on the current AQI so the feature pipeline still inserts the live row.
    """
    try:
        history = fg.read()

        if history is None or history.empty:
            return features

        history = history.copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce", utc=True)
        history = history.dropna(subset=["timestamp"])

        current_ts = pd.to_datetime(features["timestamp"], utc=True)
        history = history[
            (history["city"].astype(str).str.lower() == str(features["city"]).lower())
            & (history["timestamp"] < current_ts)
        ].sort_values("timestamp")

        if history.empty:
            return features

        recent_24 = history.tail(24)
        recent_6 = history.tail(6)

        previous_aqi = safe_float(history.iloc[-1].get("aqi"), features["aqi"])
        features["aqi_change_rate"] = float(features["aqi"] - previous_aqi)

        # Include the current AQI in live rolling values.
        features["aqi_rolling_6h"] = float(
            pd.concat([recent_6["aqi"], pd.Series([features["aqi"]])]).astype(float).mean()
        )
        features["aqi_rolling_24h"] = float(
            pd.concat([recent_24["aqi"], pd.Series([features["aqi"]])]).astype(float).mean()
        )

        print(
            "[✓] Updated live rolling features "
            f"(6h={features['aqi_rolling_6h']:.2f}, "
            f"24h={features['aqi_rolling_24h']:.2f}, "
            f"change={features['aqi_change_rate']:.2f})"
        )

    except Exception as e:
        print(f"[!] Could not calculate rolling features from Hopsworks history: {e}")
        print("[i] Using safe fallback rolling features based on current AQI.")

    return features

# HOPSWORKS STORAGE 

def store_in_hopsworks(features: dict) -> None:
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
        description="Hourly AQI + weather features (with targets)",
        event_time="timestamp",
    )

    features = update_live_trend_features(features, fg)

    df = pd.DataFrame([features])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Wait for the Hopsworks insert job so the new row is committed and visible.
    fg.insert(df, write_options={"wait_for_job": True})

    print(f"[✓] Inserted 1 row at {features['timestamp']} UTC")

#PIPELINE ENTRYPOINT 

def run() -> None:
    print("=== Feature Pipeline ===")
    print(f"Fetching data for {CITY}...")

    aqi_data = fetch_aqi_data(CITY)
    weather_data = fetch_weather_data(LAT, LON)
    air_pollution_data = fetch_air_pollution_data(LAT, LON)

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
