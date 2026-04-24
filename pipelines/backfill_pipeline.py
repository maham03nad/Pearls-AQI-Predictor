import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd

from src.config import CITY, RAW_DATA_PATH, FEATURE_DATA_PATH
from src.feature_engineering import build_features


def estimate_openweather_aqi(pm25, pm10):
    """
    OpenWeather AQI scale:
    1 = Good
    2 = Fair
    3 = Moderate
    4 = Poor
    5 = Very Poor
    """

    if pm25 <= 10 and pm10 <= 20:
        return 1
    if pm25 <= 25 and pm10 <= 50:
        return 2
    if pm25 <= 50 and pm10 <= 100:
        return 3
    if pm25 <= 75 and pm10 <= 200:
        return 4
    return 5


def create_demo_historical_karachi_data(city, start_date, end_date):
    """
    Creates demo historical Karachi weather + pollution data.

    This is used for model training when real historical API data
    is not available.
    """

    timestamps = pd.date_range(
        start=start_date,
        end=end_date,
        freq="h",
    )

    np.random.seed(42)

    rows = []

    for event_time in timestamps:
        hour = event_time.hour
        month = event_time.month

        traffic_effect = 0

        if 7 <= hour <= 10:
            traffic_effect = 18
        elif 17 <= hour <= 22:
            traffic_effect = 22

        seasonal_effect = 8 if month in [11, 12, 1, 2] else 0

        pm25 = (
            30
            + traffic_effect
            + seasonal_effect
            + np.random.normal(0, 8)
        )

        pm10 = (
            70
            + traffic_effect * 1.4
            + seasonal_effect
            + np.random.normal(0, 15)
        )

        pm25 = max(pm25, 5)
        pm10 = max(pm10, 10)

        temperature = 28 + np.random.normal(0, 4)
        humidity = 60 + np.random.normal(0, 12)
        humidity = min(max(humidity, 20), 95)

        openweather_aqi = estimate_openweather_aqi(pm25, pm10)

        row = {
            "city": city,
            "event_time": event_time,
            "weather_datetime": event_time,
            "temperature": round(temperature, 2),
            "feels_like": round(temperature + np.random.normal(1, 2), 2),
            "pressure": int(1008 + np.random.normal(0, 4)),
            "humidity": round(humidity, 2),
            "wind_speed": round(np.random.uniform(1, 8), 2),
            "wind_degree": int(np.random.uniform(0, 360)),
            "clouds": int(np.random.uniform(0, 100)),
            "rain_1h": round(max(0, np.random.normal(0.2, 0.5)), 2),
            "pollution_datetime": event_time,
            "openweather_aqi": openweather_aqi,
            "co": round(np.random.uniform(200, 1200), 2),
            "no": round(np.random.uniform(0, 20), 2),
            "no2": round(np.random.uniform(5, 80), 2),
            "o3": round(np.random.uniform(10, 120), 2),
            "so2": round(np.random.uniform(1, 40), 2),
            "pm2_5": round(pm25, 2),
            "pm10": round(pm10, 2),
            "nh3": round(np.random.uniform(1, 30), 2),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def run_backfill(city, start_date, end_date):
    print("Starting Karachi historical data backfill...")
    print(f"City: {city}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    print("Generating historical weather and pollution data...")
    raw_df = create_demo_historical_karachi_data(
        city=city,
        start_date=start_date,
        end_date=end_date,
    )

    print("Saving raw historical data...")
    raw_df.to_csv(RAW_DATA_PATH, index=False)

    print("Building historical features...")
    feature_df = build_features(raw_df)

    print("Creating training target...")
    feature_df["target_aqi_next_hour"] = feature_df["openweather_aqi"].shift(-1)
    feature_df = feature_df.dropna().reset_index(drop=True)

    print("Saving processed historical features...")
    feature_df.to_csv(FEATURE_DATA_PATH, index=False)

    print("Historical backfill completed successfully.")
    print(f"Raw data saved at: {RAW_DATA_PATH}")
    print(f"Feature data saved at: {FEATURE_DATA_PATH}")
    print(f"Total raw rows: {len(raw_df)}")
    print(f"Total feature rows: {len(feature_df)}")
    print(feature_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--city",
        default=CITY,
        help="City name, for example Karachi,PK",
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Start date, for example 2025-01-01",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="End date, for example 2025-04-01",
    )

    args = parser.parse_args()

    run_backfill(
        city=args.city,
        start_date=args.start,
        end_date=args.end,
    )
