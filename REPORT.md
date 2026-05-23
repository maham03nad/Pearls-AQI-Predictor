#  Project Report---AQI Predictor

This project predicts Air Quality Index (AQI) for Karachi using pollutant, weather, time-based, and rolling AQI features.

## System Architecture

The project collects AQI and pollutant data from AQICN and weather data from OpenWeather. The data is processed through feature engineering and stored in Hopsworks Feature Store. ML models are trained through a training pipeline and the final model is registered in Hopsworks Model Registry. The predictions are displayed through a Streamlit dashboard.

```text
AQICN + OpenWeather
        ↓
Feature Pipeline
        ↓
Hopsworks Feature Store
        ↓
Feature View / Training Data
        ↓
Training Pipeline
        ↓
Hopsworks Model Registry
        ↓
Streamlit Dashboard

```

## Data Sources

2 data sources are used:
- AQICN: AQI and pollutant data
- OpenWeather: weather data

AQICN API:

AQICN provided AQI and pollutant values:

AQI
PM2.5
PM10
O3
NO2
SO2
CO

OpenWeather API:

OpenWeather provided weather features:

Temperature
Humidity
Pressure
Wind speed
Wind direction

Both APIs are used because AQI depends on  weather conditions and  pollutant concentration.

## Feature Engineering

The project created pollutant features, weather features, time-based features, cyclic features, rolling AQI averages, and future AQI target columns.Feature Engineering

The project created contins 28 features including:

Pollutant features: PM2.5, PM10, O3, NO2, SO2, CO

Weather features: temperature, humidity, pressure, wind speed, wind direction

Time-based features: hour, day of week, month, weekend flag

Cyclical features: hour sine/cosine and month sine/cosine

Rolling features: AQI rolling average over 6 hours and 24 hours

Target columns: target_aqi_3h, target_aqi_24h, target_aqi_72h

Live feature rows store future target columns as NaN because future AQI is unknown at insertion time. Historical target values are created from past data using time-based shifting

## Hopsworks Feature Store

The engineered data was stored in Hopsworks Feature Store.

Feature Group: aqi_features

Feature View: aqi_feature_view

Training Data: version 1


## EDA

EDA was performed in `eda.ipynb`. It included missing value analysis, AQI statistics, AQI trends, pollutant relationships, correlation heatmap, and model comparison.

EDA Findings:

The dataset contained 8,589 rows and 28 engineered features.

After removing rows with missing future target values, 8,543 clean records remained.

Missing values mainly appeared in future target columns, which is expected because the latest rows do not have future AQI values available.
AQI values showed variation over time, making forecasting meaningful.

Pollutants such as PM2.5, PM10, O3, NO2, SO2, and CO were useful AQI-related features.

Weather features were included because temperature, humidity, pressure, and wind affect pollution concentration and movement.

## Model Training

Multiple regression models were trained and compared:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- LSTM

Since AQI prediction is a regression taskn so the models were evaluated using MAE, RMSE, and R² score.
The final production model was selected based on evaluation metrics and registered in Hopsworks Model Registry.

## Results

The final registered model achieved:

- MAE: 11.76
- RMSE: 19.97
- R²: 0.738

R² = 0.738 means the model explains a good portion of AQI variation and performs reasonably well for AQI forecasting.

## Model Comparison:

In the notebook model comparison, Random Forest performed best on the random train-test split.

But the production training pipeline uses  Gradient Boosting as the final registered model based on the pipeline evaluation metrics in Hopsworks Model Registry.

This difference happen because notebook experiments and production pipeline evaluation uses different split methods or evaluation settings.

## Final Model Results

The final model was registered in Hopsworks Model Registry as:

Model Name: aqi_predictor
Final Model: GradientBoost

MAE: 11.76

RMSE: 19.97

R²: 0.738

An R² score of 0.738 means that the model explains approximately 73.8% of the variation in AQI values. This indicates that the model has learned meaningful relationships between pollutant, weather, time-based, and rolling AQI features.

## Explainability

SHAP feature is used to explain global model behavior and LIME explanation is also added to explain an individual prediction.

## Deployment and Automation

The dashboard was deployed using Streamlit Cloud. GitHub Actions were used to automate the feature pipeline and training pipeline.

- Feature pipeline runs hourly.
- Training pipeline runs daily.
- Backfill pipeline can be triggered manually.

## Dashboard

The dashboard displays:

- Current AQI
- AQI category
- AQI gauge
- Current weather conditions
- Pollutant breakdown
- 72-hour AQI forecast
- 3-day forecast summary
- SHAP feature importance
- AQI scale reference
- AQI health alerts
- Aqi class(1-5)


## Limitations

- AQI can be affected by external factors not included in the dataset.
- Live rows do not contain future target values because future AQI is unknown at insertion time.
- Historical future target values are generated from past data using time-based shifting.

## Project Links

- GitHub Repository: https://github.com/maham03nad/pearls-aqi-predictor
- Live Streamlit Dashboard: https://pearls-aqi-predictor-5zku3wspc4pufnzbptlxqz.streamlit.app
- Dashboard Source Code: `streamlit.app/app.py`
- Project Report: `REPORT.md`
- EDA Notebook: `eda.ipynb`

## Conclusion

This project successfully implements an end-to-end AQI prediction system for Karachi with automated data pipelines, Hopsworks Feature Store, Hopsworks Model Registry, explainability using SHAP and LIME, and an interactive Streamlit dashboard for real-time and forecasted AQI monitoring.