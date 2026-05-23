"""
STEP 5a: FastAPI Backend
Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
import joblib
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import hopsworks
import requests


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ─── CONFIG ─────────────────────────────────────────────

HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")

CITY = os.getenv("CITY") or "karachi"

def get_float_env(name, default):
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default

LAT = get_float_env("LAT", 24.8607)
LON = get_float_env("LON", 67.0011)

HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST") or "eu-west.cloud.hopsworks.ai"
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT") or "aqi_project_10pearls"
HOPSWORKS_PORT = int(os.getenv("HOPSWORKS_PORT") or 443)

AQI_THRESHOLDS = [
    (50, "Good", "#00e400"),
    (100, "Moderate", "#ffff00"),
    (150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (200, "Unhealthy", "#ff0000"),
    (300, "Very Unhealthy", "#8f3f97"),
    (500, "Hazardous", "#7e0023"),
]

FEATURE_COLS = [
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "temp", "humidity", "pressure", "wind_speed", "wind_deg",
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "aqi_change_rate", "aqi_rolling_6h", "aqi_rolling_24h",
]

# ─── APP SETUP ──────────────────────────────────────────

app = FastAPI(title="AQI Predictor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


# ─── HELPER FUNCTIONS ───────────────────────────────────

def get_aqi_category(aqi: float):
    """Convert AQI number to category label and color."""
    for limit, label, color in AQI_THRESHOLDS:
        if aqi <= limit:
            return label, color
    return "Hazardous", "#7e0023"


def get_health_advice(category: str) -> str:
    """Return health advice based on AQI category."""
    advice = {
        "Good": "Air quality is satisfactory. Enjoy outdoor activities.",
        "Moderate": "Acceptable air quality. Unusually sensitive people should limit outdoor exertion.",
        "Unhealthy for Sensitive Groups": "Sensitive groups should reduce outdoor activity.",
        "Unhealthy": "Everyone should reduce prolonged outdoor exertion.",
        "Very Unhealthy": "Health alert — everyone should avoid outdoor activity.",
        "Hazardous": "Emergency conditions. Everyone should stay indoors.",
    }
    return advice.get(category, "Check local guidelines.")


def load_model():
    """Load model from Hopsworks Model Registry."""
    global _model

    if _model is not None:
        return _model

    if not HOPSWORKS_KEY:
        print("[!] Missing HOPSWORKS_API_KEY")
        return None

    try:
        print("Loading model from Hopsworks...")

        project = hopsworks.login(
            host=HOPSWORKS_HOST,
            port=HOPSWORKS_PORT,
            project=HOPSWORKS_PROJECT,
            api_key_value=HOPSWORKS_KEY,
        )

        mr = project.get_model_registry()
        hw_model = mr.get_model("aqi_predictor", version=1)
        model_dir = hw_model.download()

        for model_file in ["GradientBoost.pkl", "RandomForest.pkl", "Ridge.pkl"]:
            path = os.path.join(model_dir, model_file)
            if os.path.exists(path):
                _model = joblib.load(path)
                print(f"[✓] Loaded model: {model_file}")
                return _model

        print("[!] No model file found in downloaded model directory.")

    except Exception as e:
        print(f"[!] Could not load model from Hopsworks: {e}")
        _model = None

    return _model


def fetch_current_data() -> dict:
    """
    Fetch live AQI + weather data from APIs.
    Returns a flat dictionary with all required features.
    """

    if not OPENWEATHER_KEY:
        raise ValueError("Missing OPENWEATHER_KEY")

    # ── Fetch AQI from AQICN ─────────────────────────────
    try:
        if not AQICN_TOKEN:
            raise ValueError("Missing AQICN_TOKEN")

        aqi_url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
        aqi_resp = requests.get(aqi_url, timeout=10)
        aqi_resp.raise_for_status()

        aqi_json = aqi_resp.json()

        if aqi_json.get("status") != "ok":
            raise ValueError(f"AQICN error: {aqi_json}")

        iaqi = aqi_json["data"].get("iaqi", {})
        current_aqi = float(aqi_json["data"].get("aqi", 0))

        pm25 = float(iaqi.get("pm25", {}).get("v", 0))
        pm10 = float(iaqi.get("pm10", {}).get("v", 0))
        o3 = float(iaqi.get("o3", {}).get("v", 0))
        no2 = float(iaqi.get("no2", {}).get("v", 0))
        so2 = float(iaqi.get("so2", {}).get("v", 0))
        co = float(iaqi.get("co", {}).get("v", 0))

    except Exception as e:
        print(f"[!] AQICN failed, using OpenWeather Air Pollution fallback: {e}")

        ap_url = (
            f"https://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"
        )

        ap_resp = requests.get(ap_url, timeout=10)
        ap_resp.raise_for_status()

        ap_json = ap_resp.json()
        comp = ap_json["list"][0]["components"]
        aqi_raw = ap_json["list"][0]["main"]["aqi"]

        aqi_map = {1: 25, 2: 75, 3: 125, 4: 175, 5: 300}
        current_aqi = float(aqi_map.get(aqi_raw, 50))

        pm25 = float(comp.get("pm2_5", 0))
        pm10 = float(comp.get("pm10", 0))
        o3 = float(comp.get("o3", 0))
        no2 = float(comp.get("no2", 0))
        so2 = float(comp.get("so2", 0))
        co = float(comp.get("co", 0))

    # ── Fetch Weather from OpenWeatherMap ────────────────
    w_url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}&units=metric"
    )

    w_resp = requests.get(w_url, timeout=10)
    w_resp.raise_for_status()
    w_json = w_resp.json()

    temp = float(w_json["main"]["temp"])
    humidity = float(w_json["main"]["humidity"])
    pressure = float(w_json["main"]["pressure"])
    wind_speed = float(w_json["wind"]["speed"])
    wind_deg = float(w_json["wind"].get("deg", 0))

    now = datetime.utcnow()
    hour = now.hour
    day = now.weekday()
    month = now.month

    return {
        "current_aqi": current_aqi,
        "pm25": pm25,
        "pm10": pm10,
        "o3": o3,
        "no2": no2,
        "so2": so2,
        "co": co,
        "temp": temp,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "wind_deg": wind_deg,
        "hour": hour,
        "day_of_week": day,
        "month": month,
        "is_weekend": int(day >= 5),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "aqi_change_rate": 0.0,
        "aqi_rolling_6h": current_aqi,
        "aqi_rolling_24h": current_aqi,
    }


def build_feature_vector(data: dict, future_time: datetime) -> list:
    """
    Build a single feature vector for one future timestamp.
    Pollutants + weather stay same. Only time features change.
    """

    fh = future_time.hour
    fday = future_time.weekday()
    fmon = future_time.month

    return [
        data["pm25"],
        data["pm10"],
        data["o3"],
        data["no2"],
        data["so2"],
        data["co"],
        data["temp"],
        data["humidity"],
        data["pressure"],
        data["wind_speed"],
        data["wind_deg"],
        fh,
        fday,
        fmon,
        int(fday >= 5),
        float(np.sin(2 * np.pi * fh / 24)),
        float(np.cos(2 * np.pi * fh / 24)),
        float(np.sin(2 * np.pi * fmon / 12)),
        float(np.cos(2 * np.pi * fmon / 12)),
        data["aqi_change_rate"],
        data["aqi_rolling_6h"],
        data["aqi_rolling_24h"],
    ]


def predict_72_hours(current_data: dict) -> list:
    """Generate 72 hourly AQI predictions."""
    model = load_model()
    now = datetime.utcnow()
    forecasts = []

    for h in range(1, 73):
        future_time = now + timedelta(hours=h)
        features = build_feature_vector(current_data, future_time)

        if model is not None:
            pred_aqi = float(model.predict([features])[0])
        else:
            noise = np.random.normal(0, 3)
            pred_aqi = current_data["current_aqi"] + noise

        pred_aqi = max(0.0, min(500.0, pred_aqi))
        label, color = get_aqi_category(pred_aqi)

        forecasts.append({
            "timestamp": future_time.strftime("%Y-%m-%d %H:%M"),
            "hour_from_now": h,
            "aqi": round(pred_aqi, 1),
            "category": label,
            "color": color,
        })

    return forecasts


# ─── API ENDPOINTS ──────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "AQI Predictor API is running",
        "endpoints": ["/current", "/forecast", "/health", "/docs"],
    }


@app.get("/current")
def get_current_aqi():
    try:
        data = fetch_current_data()
        label, color = get_aqi_category(data["current_aqi"])
        advice = get_health_advice(label)

        return {
            "city": CITY.capitalize(),
            "aqi": data["current_aqi"],
            "category": label,
            "color": color,
            "advice": advice,
            "alert": data["current_aqi"] > 150,
            "pollutants": {
                "pm25": data["pm25"],
                "pm10": data["pm10"],
                "o3": data["o3"],
                "no2": data["no2"],
                "so2": data["so2"],
                "co": data["co"],
            },
            "weather": {
                "temp": data["temp"],
                "humidity": data["humidity"],
                "pressure": data["pressure"],
                "wind_speed": data["wind_speed"],
                "wind_deg": data["wind_deg"],
            },
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast")
def get_forecast():
    try:
        current = fetch_current_data()
        predictions = predict_72_hours(current)

        days_dict = {}
        for p in predictions:
            day_key = p["timestamp"][:10]
            days_dict.setdefault(day_key, []).append(p)

        daily = []
        for date, hours in days_dict.items():
            aqis = [h["aqi"] for h in hours]
            avg_aqi = round(float(np.mean(aqis)), 1)
            label, color = get_aqi_category(avg_aqi)

            daily.append({
                "date": date,
                "avg_aqi": avg_aqi,
                "max_aqi": round(max(aqis), 1),
                "min_aqi": round(min(aqis), 1),
                "category": label,
                "color": color,
                "advice": get_health_advice(label),
                "hourly": hours,
            })

        return {
            "city": CITY.capitalize(),
            "days": daily,
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    model = load_model()

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "city": CITY,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }