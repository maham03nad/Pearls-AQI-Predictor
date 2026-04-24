import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse

from src.config import CITY, FEATURE_DATA_PATH
from src.model_training import train_aqi_models


def run_training(city):
    print("Starting Karachi AQI model training pipeline...")
    print(f"City: {city}")
    print(f"Feature file: {FEATURE_DATA_PATH}")

    metrics = train_aqi_models(FEATURE_DATA_PATH)

    print("Training pipeline completed successfully.")
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--city",
        default=CITY,
        help="City name, for example Karachi,PK",
    )

    args = parser.parse_args()

    run_training(city=args.city)
