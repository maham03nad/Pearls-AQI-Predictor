"""
 FastAPI Backend
Run: uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""
import os
import joblib
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import hopsworks
import requests
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG ================= #

HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")

CITY = os.getenv("CITY") or "karachi"

LAT = float(os.getenv("LAT", 24.8607))
LON = float(os.getenv("LON", 67.0011))

MODEL_CACHE_PATH = "aqi_model.pkl"

AQI_THRESHOLDS = [
    (50, "Good", "#00e400"),
    (100, "Moderate", "#ffff00"),
    (150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (200, "Unhealthy", "#ff0000"),
    (300, "Very Unhealthy", "#8f3f97"),
    (500, "Hazardous", "#7e0023"),
]

_model = None


# ================= MODEL LOAD (FIXED) ================= #

def load_model():
    global _model

    if _model is not None:
        return _model

    # ✅ 1. Local cache FIRST (FAST)
    if os.path.exists(MODEL_CACHE_PATH):
        print("⚡ Loading model from local cache")
        _model = joblib.load(MODEL_CACHE_PATH)
        return _model

    # ❌ 2. If not cached → download once
    try:
        print("📥 Downloading model from Hopsworks...")

        project = hopsworks.login(
            host="eu-west.cloud.hopsworks.ai",
            project="aqi_project_10pearls",
            api_key_value=HOPSWORKS_KEY,
        )

        mr = project.get_model_registry()
        model_obj = mr.get_model("aqi_predictor", version=1)
        path = model_obj.download()

        for f in ["GradientBoost.pkl", "RandomForest.pkl", "Ridge.pkl"]:
            fp = os.path.join(path, f)
            if os.path.exists(fp):
                model = joblib.load(fp)

                # ✅ SAVE LOCALLY (IMPORTANT FIX)
                joblib.dump(model, MODEL_CACHE_PATH)

                _model = model
                print("✅ Model downloaded & cached")
                return _model

    except Exception as e:
        print("❌ Model load failed:", e)

    return None


# ================= FASTAPI LIFESPAN ================= #

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = load_model()
    yield

app = FastAPI(title="AQI Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= HELPERS ================= #

def get_aqi_category(aqi):
    for limit, label, color in AQI_THRESHOLDS:
        if aqi <= limit:
            return label, color
    return "Hazardous", "#7e0023"


@lru_cache(maxsize=1)
def cached_fetch():
    return fetch_current_data()


# ================= DATA FETCH ================= #

def fetch_current_data():
    try:
        url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
        res = requests.get(url, timeout=10).json()

        if res.get("status") == "ok":
            iaqi = res["data"].get("iaqi", {})
            current_aqi = float(res["data"].get("aqi", 0))

            def g(k):
                return float(iaqi.get(k, {}).get("v", 0))

            pm25, pm10, o3, no2, so2, co = map(
                g, ["pm25", "pm10", "o3", "no2", "so2", "co"]
            )
        else:
            raise Exception("AQICN failed")

    except:
        url = (
            f"https://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"
        )
        res = requests.get(url, timeout=10).json()

        comp = res["list"][0]["components"]
        aqi_raw = res["list"][0]["main"]["aqi"]

        current_aqi = {1: 25, 2: 75, 3: 125, 4: 175, 5: 300}.get(aqi_raw, 50)

        pm25 = comp.get("pm2_5", 0)
        pm10 = comp.get("pm10", 0)
        o3 = comp.get("o3", 0)
        no2 = comp.get("no2", 0)
        so2 = comp.get("so2", 0)
        co = comp.get("co", 0)

    weather = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}&units=metric"
    ).json()

    now = datetime.utcnow()

    return {
        "current_aqi": current_aqi,
        "pm25": pm25,
        "pm10": pm10,
        "o3": o3,
        "no2": no2,
        "so2": so2,
        "co": co,
        "temp": weather["main"]["temp"],
        "humidity": weather["main"]["humidity"],
        "pressure": weather["main"]["pressure"],
        "wind_speed": weather["wind"]["speed"],
        "wind_deg": weather["wind"].get("deg", 0),
        "hour": now.hour,
        "day_of_week": now.weekday(),
        "month": now.month,
        "is_weekend": int(now.weekday() >= 5),
        "aqi_change_rate": 0.0,
        "aqi_rolling_6h": current_aqi,
        "aqi_rolling_24h": current_aqi,
    }


# ================= FEATURES ================= #

def build_feature_vector(d, t):
    return [
        d["pm25"], d["pm10"], d["o3"], d["no2"], d["so2"], d["co"],
        d["temp"], d["humidity"], d["pressure"], d["wind_speed"], d["wind_deg"],
        t.hour,
        t.weekday(),
        t.month,
        int(t.weekday() >= 5),
        np.sin(2*np.pi*t.hour/24),
        np.cos(2*np.pi*t.hour/24),
        np.sin(2*np.pi*t.month/12),
        np.cos(2*np.pi*t.month/12),
        d["aqi_change_rate"],
        d["aqi_rolling_6h"],
        d["aqi_rolling_24h"],
    ]


# ================= PREDICT ================= #

def predict_72h(data):
    model = _model
    now = datetime.utcnow()
    out = []

    for i in range(1, 73):
        t = now + timedelta(hours=i)
        x = build_feature_vector(data, t)

        if model:
            y = float(model.predict([x])[0])
        else:
            y = data["current_aqi"]

        label, color = get_aqi_category(y)

        out.append({
            "time": t.strftime("%Y-%m-%d %H:%M"),
            "aqi": round(y, 1),
            "category": label,
            "color": color
        })

    return out


# ================= ROUTES ================= #

@app.get("/current")
def current():
    try:
        d = cached_fetch()
        label, color = get_aqi_category(d["current_aqi"])
        return {
            "city": CITY,
            "aqi": d["current_aqi"],
            "category": label,
            "color": color
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/forecast")
def forecast():
    try:
        d = cached_fetch()
        return predict_72h(d)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None
    }