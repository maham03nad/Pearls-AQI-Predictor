# Pearls AQI Predictor - Karachi

This project predicts the Air Quality Index AQI for Karachi, Pakistan using a serverless machine learning pipeline.

The system fetches weather and pollutant data from OpenWeather API, creates machine learning features, stores processed feature data, trains AQI prediction models, and provides predictions through a Flask API and Streamlit dashboard.

---

## City

Karachi, Pakistan

Latitude: `24.8607`  
Longitude: `67.0011`

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- OpenWeather API
- Hopsworks Feature Store optional
- GitHub Actions
- Flask
- Streamlit
- GitHub

---

## Project Features

### 1. Feature Pipeline Development

The feature pipeline fetches raw weather and air pollution data for Karachi.

It collects:

- Temperature
- Feels like temperature
- Pressure
- Humidity
- Wind speed
- Wind direction
- Cloud coverage
- Rain
- CO
- NO
- NO2
- O3
- SO2
- PM2.5
- PM10
- NH3
- OpenWeather AQI

It then computes features such as:

- Hour
- Day
- Month
- Day of week
- Weekend flag
- AQI change rate
- PM2.5 change rate
- PM10 change rate
- Rolling averages
- Lag features

The processed features are saved locally in:

```text
data/features/karachi_feature_pipeline.csv
