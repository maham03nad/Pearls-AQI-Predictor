import pandas as pd


def build_features(raw_df):
    df = raw_df.copy()

    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)

    # Time-based features
    df["hour"] = df["event_time"].dt.hour
    df["day"] = df["event_time"].dt.day
    df["month"] = df["event_time"].dt.month
    df["day_of_week"] = df["event_time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Derived AQI and pollutant change features
    df["aqi_change_rate"] = df["openweather_aqi"].diff().fillna(0)
    df["pm25_change_rate"] = df["pm2_5"].diff().fillna(0)
    df["pm10_change_rate"] = df["pm10"].diff().fillna(0)

    # Rolling average features
    df["pm25_rolling_3"] = df["pm2_5"].rolling(window=3, min_periods=1).mean()
    df["pm10_rolling_3"] = df["pm10"].rolling(window=3, min_periods=1).mean()
    df["aqi_rolling_3"] = df["openweather_aqi"].rolling(window=3, min_periods=1).mean()

    df["pm25_rolling_6"] = df["pm2_5"].rolling(window=6, min_periods=1).mean()
    df["pm10_rolling_6"] = df["pm10"].rolling(window=6, min_periods=1).mean()
    df["aqi_rolling_6"] = df["openweather_aqi"].rolling(window=6, min_periods=1).mean()

    # Lag features
    df["aqi_lag_1"] = df["openweather_aqi"].shift(1).fillna(df["openweather_aqi"])
    df["pm25_lag_1"] = df["pm2_5"].shift(1).fillna(df["pm2_5"])
    df["pm10_lag_1"] = df["pm10"].shift(1).fillna(df["pm10"])

    feature_columns = [
        "city",
        "event_time",
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

    feature_df = df[feature_columns].copy()

    # Hopsworks works better with timezone-naive datetime
    feature_df["event_time"] = pd.to_datetime(feature_df["event_time"]).dt.tz_localize(None)

    return feature_df
