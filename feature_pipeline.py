"""
STEP 1: Feature Pipeline
==============================================
- Fetch live AQI/pollutant data from AQICN
- Fetch weather + fallback pollutant data from OpenWeather
- Engineer model-ready features
- Calculate live rolling AQI features using recent Hopsworks history
- Store the final live row in Hopsworks Feature Store
Run:
    python feature_pipeline.py
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, Tuple
import hopsworks
import json
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass
 
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

CACHE_FILE = "last_known_values.json"
POLLUTANT_KEYS = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co"]

# City-level safe defaults are used only if API + cache fail.
# These prevent zero-filled pollutant values from confusing the model.
CITY_DEFAULTS = {
    "pm25": 35.0,
    "pm10": 80.0,
    "o3": 50.0,
    "no2": 20.0,
    "so2": 5.0,
    "co": 300.0,
    "aqi": 100.0,
}

#  CACHE HELPERS 

def load_cache() -> Dict[str, float]:
    """Load last-known valid values from local JSON cache."""
    if not os.path.exists(CACHE_FILE):
        return {}

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: float(v) for k, v in data.items() if v is not None}
    except Exception as exc:
        print(f"[!] Cache read failed: {exc}")
        return {}


def save_cache(data: Dict[str, float]) -> None:
    """Save valid positive AQI/pollutant values for future fallback use."""
    cache = load_cache()

    for key in POLLUTANT_KEYS:
        value = data.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if value > 0:
            cache[key] = value

    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        preview = {k: round(v, 2) for k, v in cache.items()}
        print(f"[✓] Cache updated: {preview}")
    except Exception as exc:
        print(f"[!] Cache save failed: {exc}")


def get_cached_or_default(key: str) -> float:
    """Return cached value first, otherwise city-level default."""
    cache = load_cache()
    cached = cache.get(key)

    try:
        cached = float(cached)
    except (TypeError, ValueError):
        cached = 0.0

    if cached > 0:
        return cached

    return float(CITY_DEFAULTS.get(key, 0.0))

#  API FETCH FUNCTIONS 

def fetch_aqi_data(city: str) -> dict:
    """Fetch current AQI and AQI pollutant values from AQICN."""
    if not AQICN_TOKEN:
        raise ValueError("Missing AQICN_TOKEN")

    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    payload = response.json()
    if payload.get("status") != "ok":
        raise ValueError(f"AQICN error: {payload}")

    return payload["data"]

def fetch_openweather_pollution() -> dict:
    """
    Fetch pollutant components from OpenWeather Air Pollution API.
    This is used as a fallback when AQICN data looks suspicious or is missing.
    """
    if not OPENWEATHER_KEY:
        print("[!] Missing OPENWEATHER_KEY, skipping OpenWeather pollution fallback.")
        return {}

    try:
        url = (
            "https://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        components = response.json()["list"][0]["components"]

        result = {
            "pm25": float(components.get("pm2_5", 0) or 0),
            "pm10": float(components.get("pm10", 0) or 0),
            "o3": float(components.get("o3", 0) or 0),
            "no2": float(components.get("no2", 0) or 0),
            "so2": float(components.get("so2", 0) or 0),
            "co": float(components.get("co", 0) or 0),
        }
        print(f"[✓] OpenWeather pollution: {result}")
        return result

    except Exception as exc:
        print(f"[!] OpenWeather pollution fetch failed: {exc}")
        return {}

def fetch_weather_data(lat: float, lon: float) -> dict:
    """Fetch current weather values from OpenWeatherMap."""
    if not OPENWEATHER_KEY:
        raise ValueError("Missing OPENWEATHER_KEY")

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

#  POLLUTANT RESOLUTION 

def safe_positive(value) -> float:
    """Convert a value to float only if it is positive otherwise return it 0."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if np.isnan(value) or value <= 0:
        return 0.0

    return value

def resolve_value(key: str, aqicn_value: float, ow_value: float) -> Tuple[float, str]:
    """Pick AQICN → OpenWeather → cache/default in that order."""
    if aqicn_value > 0:
        return aqicn_value, "AQICN"

    if ow_value > 0:
        return ow_value, "OpenWeather"

    fallback = get_cached_or_default(key)
    source = "cache/default"
    return fallback, source

def resolve_pollutants(aqicn_data: dict, ow_pollution: dict) -> dict:

    iaqi = aqicn_data.get("iaqi", {})
    overall_aqi = safe_positive(aqicn_data.get("aqi", 0))

    def aqicn_pollutant(key: str) -> float:
        return safe_positive(iaqi.get(key, {}).get("v", None))

    def ow_pollutant(key: str) -> float:
        return safe_positive(ow_pollution.get(key, 0))

    aqicn_values = {
        "pm25": aqicn_pollutant("pm25"),
        "pm10": aqicn_pollutant("pm10"),
        "o3": aqicn_pollutant("o3"),
        "no2": aqicn_pollutant("no2"),
        "so2": aqicn_pollutant("so2"),
        "co": aqicn_pollutant("co"),
    }

    ow_values = {
        "pm25": ow_pollutant("pm25"),
        "pm10": ow_pollutant("pm10"),
        "o3": ow_pollutant("o3"),
        "no2": ow_pollutant("no2"),
        "so2": ow_pollutant("so2"),
        "co": ow_pollutant("co"),
    }

    pollutants = {}
    sources = {}

    # If AQICN PM2.5 is equal to AQI and OpenWeather has a positive PM2.5 value, it's likely that AQICN is using PM2.5 as a fallback for overall AQI.
    # use OpenWeather to avoid repeated same-value rows in live data.
    pm25_suspicious = (
        aqicn_values["pm25"] > 0
        and overall_aqi > 0
        and abs(aqicn_values["pm25"] - overall_aqi) < 2.0
        and ow_values["pm25"] > 0
    )

    if pm25_suspicious:
        pollutants["pm25"] = ow_values["pm25"]
        sources["pm25"] = "OpenWeather (AQICN pm25≈AQI fallback)"
        print(
            f"[~] PM2.5 fallback: AQICN pm25={aqicn_values['pm25']} "
            f"≈ AQI={overall_aqi}; using OpenWeather pm25={ow_values['pm25']}"
        )
    else:
        pollutants["pm25"], sources["pm25"] = resolve_value(
            "pm25", aqicn_values["pm25"], ow_values["pm25"]
        )
    # Use OpenWeather when AQICN is missing.

    pollutants["pm10"], sources["pm10"] = resolve_value(
        "pm10", aqicn_values["pm10"], ow_values["pm10"]
    )
    if aqicn_values["pm10"] == 0 and ow_values["pm10"] > 0:
        print(f"[~] PM10 fallback: AQICN missing; using OpenWeather pm10={ow_values['pm10']}")

    for key in ["o3", "no2", "so2", "co"]:
        pollutants[key], sources[key] = resolve_value(
            key, aqicn_values[key], ow_values[key]
        )

    print(
        "[✓] Pollutant sources: "
        + " | ".join(
            f"{key}={sources[key]}({pollutants[key]:.2f})"
            for key in ["pm25", "pm10", "o3", "no2", "so2", "co"]
        )
    )

    return pollutants

#  HOPSWORKS HISTORY + ROLLING FEATURES 

def fetch_recent_history(fg, city: str, hours: int = 24) -> pd.DataFrame:
    """Fetch recent rows from Hopsworks to compute live rolling AQI features."""
    try:
        df = fg.read()

        if df.empty:
            print("[!] Hopsworks history is empty.")
            return pd.DataFrame()

        df = df[df["city"].astype(str).str.lower() == city.lower()].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp", "aqi"])

        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=hours)
        df = df[df["timestamp"] >= cutoff].sort_values("timestamp", ascending=True)

        print(f"[✓] Fetched {len(df)} history rows from Hopsworks (last {hours}h)")
        return df

    except Exception as exc:
        print(f"[!] History fetch failed: {exc}")
        return pd.DataFrame()


def compute_rolling_features(history_df: pd.DataFrame, current_aqi: float) -> dict:
    now_ts = pd.Timestamp.utcnow()

    if history_df.empty:
        cached_aqi = get_cached_or_default("aqi")
        print("[!] No recent history available; using cache/default fallback for rolling.")
        return {
            "aqi_change_rate": round(current_aqi - cached_aqi, 4),
            "aqi_rolling_6h": round(current_aqi, 4),
            "aqi_rolling_24h": round(current_aqi, 4),
        }

    hist = history_df[["timestamp", "aqi"]].copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce", utc=True)
    hist["aqi"] = pd.to_numeric(hist["aqi"], errors="coerce")
    hist = hist.dropna(subset=["timestamp", "aqi"]).sort_values("timestamp")

    now_row = pd.DataFrame([{"timestamp": now_ts, "aqi": current_aqi}])
    combined = pd.concat([hist, now_row], ignore_index=True).sort_values("timestamp")
    aqi_series = combined["aqi"].astype(float)

    if len(aqi_series) >= 2:
        change_rate = float(aqi_series.iloc[-1] - aqi_series.iloc[-2])
    else:
        change_rate = 0.0

    rolling_6h = float(aqi_series.tail(6).mean())
    rolling_24h = float(aqi_series.tail(24).mean())

    print(
        f"[✓] Rolling features: change={change_rate:.4f}, "
        f"6h_avg={rolling_6h:.4f}, 24h_avg={rolling_24h:.4f}"
    )

    return {
        "aqi_change_rate": round(change_rate, 4),
        "aqi_rolling_6h": round(rolling_6h, 4),
        "aqi_rolling_24h": round(rolling_24h, 4),
    }

#  FEATURE ENGINEERING 

def engineer_features(
    aqi_data: dict,
    weather_data: dict,
    pollutants: dict,
    rolling: dict,
) -> dict:
    now = datetime.utcnow().replace(microsecond=0)

    overall_aqi = safe_positive(aqi_data.get("aqi", 0))
    if overall_aqi <= 0:
        overall_aqi = get_cached_or_default("aqi")

    temp = float(weather_data["main"]["temp"])
    humidity = float(weather_data["main"]["humidity"])
    pressure = float(weather_data["main"]["pressure"])
    wind_speed = float(weather_data["wind"]["speed"])
    wind_deg = float(weather_data["wind"].get("deg", 0) or 0)

    hour = now.hour
    day = now.weekday()
    month = now.month

    save_cache({**pollutants, "aqi": overall_aqi})

    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "city": CITY.capitalize(),
        "aqi": float(overall_aqi),
        "pm25": float(pollutants["pm25"]),
        "pm10": float(pollutants["pm10"]),
        "o3": float(pollutants["o3"]),
        "no2": float(pollutants["no2"]),
        "so2": float(pollutants["so2"]),
        "co": float(pollutants["co"]),
        "temp": temp,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "wind_deg": wind_deg,
        "hour": int(hour),
        "day_of_week": int(day),
        "month": int(month),
        "is_weekend": int(day >= 5),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "aqi_change_rate": float(rolling["aqi_change_rate"]),
        "aqi_rolling_6h": float(rolling["aqi_rolling_6h"]),
        "aqi_rolling_24h": float(rolling["aqi_rolling_24h"]),
        # Live future targets are unknown,to avoid leakage keeping them None.
        "target_aqi_3h": None,
        "target_aqi_24h": None,
        "target_aqi_72h": None,
    }

#  HOPSWORKS STORE 

def connect_hopsworks():
    """Login to Hopsworks and return feature group."""
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
        description="Hourly AQI + weather features with forecasting targets",
        event_time="timestamp",
    )

    return fg


def insert_features(fg, features: dict) -> None:
    """Insert one row into Hopsworks and wait for completion."""
    df = pd.DataFrame([features])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    print("[~] Final row preview:")
    print(
        df[
            [
                "timestamp",
                "city",
                "aqi",
                "pm25",
                "pm10",
                "temp",
                "aqi_change_rate",
                "aqi_rolling_6h",
                "aqi_rolling_24h",
                "target_aqi_3h",
                "target_aqi_24h",
                "target_aqi_72h",
            ]
        ].to_string(index=False)
    )
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"[✓] Inserted 1 row at {features['timestamp']}")


def run() -> None:
    print("=" * 60)
    print(" Feature Pipeline — Fixed Live Data Version")
    print("=" * 60)
    print(f"City: {CITY} | Lat: {LAT} | Lon: {LON}")
    print()

    print("▶ Step 1: Connecting to Hopsworks...")
    fg = connect_hopsworks()
    print("[✓] Hopsworks connected")
    print()

    print("▶ Step 2: Fetching recent Hopsworks history...")
    history_df = fetch_recent_history(fg, CITY, hours=24)
    print()

    print("▶ Step 3: Fetching live AQI from AQICN...")
    aqi_data = fetch_aqi_data(CITY)
    current_aqi = safe_positive(aqi_data.get("aqi", 0))
    print(f"[✓] AQICN AQI = {current_aqi}")
    print()

    print("▶ Step 4: Fetching OpenWeather pollutant fallback...")
    ow_pollution = fetch_openweather_pollution()
    print()

    print("▶ Step 5: Resolving final pollutant values...")
    pollutants = resolve_pollutants(aqi_data, ow_pollution)
    print()

    print("▶ Step 6: Fetching weather data...")
    weather_data = fetch_weather_data(LAT, LON)
    print(
        f"[✓] Weather: temp={weather_data['main']['temp']}°C, "
        f"humidity={weather_data['main']['humidity']}%, "
        f"wind={weather_data['wind']['speed']} m/s"
    )
    print()

    print("▶ Step 7: Computing rolling features...")
    rolling = compute_rolling_features(history_df, current_aqi)
    print()

    print("▶ Step 8: Engineering feature row...")
    features = engineer_features(aqi_data, weather_data, pollutants, rolling)
    print(
        f"[✓] Row ready: AQI={features['aqi']}, "
        f"PM2.5={features['pm25']}, PM10={features['pm10']}, "
        "targets=None/NaN"
    )
    print()

    print("▶ Step 9: Inserting into Hopsworks Feature Store...")
    insert_features(fg, features)
    print()

    print("=" * 60)
    print(" Feature Pipeline completed successfully ✅")
    print("=" * 60)

if __name__ == "__main__":
    run()
