import requests
import pandas as pd

from src.config import OPENWEATHER_API_KEY


def fetch_current_weather(lat, lon):
    if not OPENWEATHER_API_KEY:
        raise ValueError("OPENWEATHER_API_KEY is missing. Add it in your .env file or GitHub Secrets.")

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    weather_row = {
        "weather_datetime": pd.to_datetime(data.get("dt"), unit="s"),
        "temperature": data.get("main", {}).get("temp"),
        "feels_like": data.get("main", {}).get("feels_like"),
        "pressure": data.get("main", {}).get("pressure"),
        "humidity": data.get("main", {}).get("humidity"),
        "wind_speed": data.get("wind", {}).get("speed"),
        "wind_degree": data.get("wind", {}).get("deg", 0),
        "clouds": data.get("clouds", {}).get("all", 0),
        "rain_1h": data.get("rain", {}).get("1h", 0),
    }

    return weather_row


def fetch_current_pollution(lat, lon):
    if not OPENWEATHER_API_KEY:
        raise ValueError("OPENWEATHER_API_KEY is missing. Add it in your .env file or GitHub Secrets.")

    url = "https://api.openweathermap.org/data/2.5/air_pollution"

    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    item = data["list"][0]
    components = item.get("components", {})

    pollution_row = {
        "pollution_datetime": pd.to_datetime(item.get("dt"), unit="s"),
        "openweather_aqi": item.get("main", {}).get("aqi"),
        "co": components.get("co"),
        "no": components.get("no"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "nh3": components.get("nh3"),
    }

    return pollution_row


def fetch_karachi_raw_row(city, lat, lon):
    weather = fetch_current_weather(lat, lon)
    pollution = fetch_current_pollution(lat, lon)

    row = {
        "city": city,
        "event_time": pollution["pollution_datetime"],
        **weather,
        **pollution,
    }

    return pd.DataFrame([row])
