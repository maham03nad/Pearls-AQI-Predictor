import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import CITY, LAT, LON, RAW_DATA_PATH, FEATURE_DATA_PATH
from src.data_fetching import fetch_karachi_raw_row
from src.feature_engineering import build_features
from src.feature_store import store_features_in_hopsworks


def append_raw_data(new_raw_df):
    """
    Adds the newest raw weather + pollution row to the local raw CSV file.
    If the file already exists, old data and new data are combined.
    Duplicate city + event_time rows are removed.
    """

    if RAW_DATA_PATH.exists():
        old_raw_df = pd.read_csv(RAW_DATA_PATH)
        old_raw_df["event_time"] = pd.to_datetime(old_raw_df["event_time"])

        combined_df = pd.concat([old_raw_df, new_raw_df], ignore_index=True)

        combined_df = combined_df.drop_duplicates(
            subset=["city", "event_time"],
            keep="last",
        )
    else:
        combined_df = new_raw_df

    combined_df = combined_df.sort_values("event_time").reset_index(drop=True)
    combined_df.to_csv(RAW_DATA_PATH, index=False)

    return combined_df


def run_feature_pipeline():
    print("Starting Karachi AQI feature pipeline...")

    print("Fetching raw weather and pollutant data from OpenWeather...")
    new_raw_df = fetch_karachi_raw_row(
        city=CITY,
        lat=LAT,
        lon=LON,
    )

    print("Saving raw data locally...")
    full_raw_df = append_raw_data(new_raw_df)

    print("Computing features...")
    feature_df = build_features(full_raw_df)

    print("Saving processed features locally...")
    feature_df.to_csv(FEATURE_DATA_PATH, index=False)

    latest_feature_row = feature_df.tail(1).copy()

    print("Sending latest feature row to Hopsworks Feature Store...")
    store_features_in_hopsworks(latest_feature_row)

    print("Feature pipeline completed successfully.")
    print(f"Raw data saved at: {RAW_DATA_PATH}")
    print(f"Features saved at: {FEATURE_DATA_PATH}")
    print("Latest feature row:")
    print(latest_feature_row)


if __name__ == "__main__":
    run_feature_pipeline()
