#  Project Report---AQI Predictor
72-Hour Air Quality Forecasting System — Karachi, Pakistan

SUBMITTED BY: MAHAM NADEEM

**SUMMARY**

Pearls AQI Predictor is a fully automated, serverless ML system that forecasts the Air Quality Index (AQI) for Karachi, Pakistan 72 hours into the future. The system collects real-time pollutant and weather data, engineers meaningful features, trains multiple regression models, and serves live predictions through a Streamlit dashboard — all orchestrated via GitHub Actions CI/CD.

| Best Model      | RandomForest|Auto-selected by lowest RMSE|
| R² Score        |0.894        |89.4% variance explained    |
| RMSE            | 20.32 AQI   |Average prediction error    |
| MAE             | 11.53 AQI   |Mean absolute deviation     |
|Forecast Horizon | 72 hours    |3-day ahead prediction      |

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

In this 2 data sources are used:
- AQICN: AQI and pollutant data
- OpenWeather: weather data

AQICN API:

AQICN provided AQI and pollutant values:


| Pollutant       | Description                                    |
| --------------- | ---------------                                |
| AQI             | Overall Air Quality Index                      |
| PM2.5           | Fine particulate matter (≤2.5μm) — most harmful|       
|PM10             |Coarse particulate matter (≤10μm)               |
|O3               |Ground-level ozone                              |
|NO2              |Nitrogen dioxide — traffic/industrial           |
|SO2              | Sulphur dioxide — combustion                   |       
|CO               |Carbon monoxide                                 |

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

## Cyclical Encoding

Time features (hour, month) are encoded using sine and cosine transformations to capture their cyclical nature. Without this, a model would treat hour 23 and hour 0 as far apart when they are actually adjacent.
Formula: sin(2π × hour / 24) and cos(2π × hour / 24)

## Rolling Features

AQI rolling averages over 6-hour and 24-hour windows capture the recent trend and momentum of air quality. The AQI change rate captures the current direction of change (improving or deteriorating).

 ## Target Engineering

Future targets are created by shifting the AQI column forward. Live rows have None/NaN for target columns, as future AQI is unknown at collection time — this prevents data leakage.

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

## Model Training & Evaluation

Multiple regression models were trained and compared:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- LSTM

Since AQI prediction is a regression taskn so the models were evaluated using MAE, RMSE, and R² score.
The final production model was selected based on evaluation metrics and registered in Hopsworks Model Registry.

## Model Comparison Results

| MODEL         |  RMSE    | MAE  | R²    |
| ------------- | ---------|------|-------|
| RandomForest  |    20.32 | 11.53| 0.894 |
| GradientBoost |    40.59 | 29.59| 0.578 |
|Ridge          |    54.25 | 40.88| 0.246 |   
|LSTM           |    62.28 | 44.84| -0.01 |

RandomForest achieved the best performance with R² = 0.894, meaning it explains 89.4% of the variance in 72-hour AQI values. It is automatically registered as the production model in Hopsworks Model Registry.

## Model Selection Logic:

The training pipeline auto-selects the best model by lowest RMSE across all trained sklearn models. This prevents hardcoded model selection and ensures the pipeline adapts as data evolves.

## LSTM Experimental Model:

An LSTM (Long Short-Term Memory) neural network was trained as an experimental comparison model. It uses 24-step look-back sequences and EarlyStopping to prevent overfitting. The LSTM is not used for production predictions as it requires sequential input that is unavailable at live inference time.


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

## AQI Health Alert System

| AQI Range     |  Category            | Alert       | 
| ------------- | -------------------- |------------ |
| 0–50          | Good                 |✅ Safe      | 
| 51–100        | Moderate             |🟡 Acceptable|
|101–150        |Unhealthy (Sensitive) |🟠 Caution   | 
|151–200        |Unhealthy             |🔴 Warning   | 
|201–300        |Very Unhealthy        |🟣 Danger    |
|301+           |Hazardous             |⛔ Emergency |
## FastAPI Backend

The project also include FastAPI backend in `api.py`. 
It exposes:

- `/current` returns live AQI, pollutant, weather, alert, and timestamp data.
- `/forecast` returns a 72-hour AQI forecast with hourly predictions and daily summaries.
- `/health` verifies whether the API and model are running.
- `/docs` provides Swagger API documentation.

The live dashboard is deployed using Streamlit Cloud, while the FastAPI backend can be run locally using:

```bash
uvicorn api:app --reload
```
--- 

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

The Pearls AQI Predictor successfully delivers an end-to-end, production-grade ML system for 72-hour AQI forecasting in Karachi. The system achieves:
• R² = 0.894 on the 72-hour forecast target — strong predictive accuracy
•Fully automated hourly data collection and daily retraining via GitHub Actions
•Robust data quality handling with multi-tier API fallback strategy
•Model explainability through SHAP (global) and LIME (local) analysis
•Live public dashboard with real-time AQI insights and health alerts
•Production MLOps architecture using Hopsworks Feature Store and Model Registry

