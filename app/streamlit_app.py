import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import streamlit as st

from src.config import CITY, LAT, LON
from src.data_fetching import fetch_karachi_raw_row
from src.prediction import predict_latest_aqi


METRICS_PATH = Path("models") / "karachi_metrics.json"


st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
)

st.title("🌫️ Pearls AQI Predictor - Karachi")
st.write("This dashboard predicts the next AQI level for Karachi using weather and pollution data.")

st.sidebar.header("Project Information")
st.sidebar.write(f"City: {CITY}")
st.sidebar.write(f"Latitude: {LAT}")
st.sidebar.write(f"Longitude: {LON}")

if st.button("Predict Karachi AQI"):
    try:
        raw_df = fetch_karachi_raw_row(
            city=CITY,
            lat=LAT,
            lon=LON,
        )

        prediction_df = predict_latest_aqi(raw_df)
        prediction = prediction_df.iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Current OpenWeather AQI",
            int(prediction["openweather_aqi"]),
        )

        col2.metric(
            "Predicted Next AQI",
            round(prediction["predicted_aqi_next_hour"], 2),
        )

        col3.metric(
            "AQI Category",
            prediction["category_name"],
        )

        st.subheader("Latest Prediction Data")
        st.dataframe(prediction_df)

        st.subheader("Pollutant Levels")
        pollutant_data = prediction_df[["pm2_5", "pm10"]]
        st.bar_chart(pollutant_data)

        if prediction["predicted_aqi_category"] >= 4:
            st.error("Alert: Karachi AQI is predicted to be Poor or Very Poor.")
        elif prediction["predicted_aqi_category"] == 3:
            st.warning("Warning: Karachi AQI is predicted to be Moderate.")
        else:
            st.success("Karachi AQI is predicted to be Good or Fair.")

    except Exception as error:
        st.error(str(error))
        st.info("Make sure you have trained the model first.")

if METRICS_PATH.exists():
    st.subheader("Model Metrics")

    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    st.json(metrics)
else:
    st.info("Model metrics are not available yet. Run the training pipeline first.")
