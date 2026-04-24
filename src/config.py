import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURE_DIR = DATA_DIR / "features"

RAW_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

CITY = os.getenv("CITY", "Karachi,PK")
LAT = float(os.getenv("LAT", "24.8607"))
LON = float(os.getenv("LON", "67.0011"))

HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

RAW_DATA_PATH = RAW_DIR / "karachi_raw_weather_pollution.csv"
FEATURE_DATA_PATH = FEATURE_DIR / "karachi_feature_pipeline.csv"

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 1
