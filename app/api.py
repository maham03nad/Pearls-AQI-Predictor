import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify

from src.config import CITY, LAT, LON
from src.data_fetching import fetch_karachi_raw_row
from src.prediction import predict_latest_aqi


app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({
        "project": "Pearls AQI Predictor",
        "city": CITY,
        "message": "Use /predict to get Karachi AQI prediction",
    })


@app.route("/predict")
def predict():
    try:
        raw_df = fetch_karachi_raw_row(
            city=CITY,
            lat=LAT,
            lon=LON,
        )

        prediction_df = predict_latest_aqi(raw_df)
        result = prediction_df.to_dict(orient="records")[0]

        return jsonify({
            "city": CITY,
            "prediction": result,
        })

    except Exception as error:
        return jsonify({
            "error": str(error),
            "hint": "Make sure the model is trained and API key is configured.",
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
