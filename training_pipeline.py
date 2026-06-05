"""

STEP 3: Training Pipeline
 
This pipeline:
- Loads AQI features from Hopsworks Feature Store
- Uses target_aqi_72h as the final forecasting target
- Trains Random Forest, Gradient Boosting, Ridge Regression, and LSTM
- Evaluates models using RMSE, MAE, and R²
- Saves model artifacts locally
- Generates SHAP and LIME explanations
- Registers BEST model (lowest RMSE) in Hopsworks Model Registry
 
Run:
    python training_pipeline.py
"""
 
import os
import json
import joblib
import numpy as np
import pandas as pd
import hopsworks
import shap
import matplotlib
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from dotenv import load_dotenv
from lime.lime_tabular import LimeTabularExplainer
 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
 
 
# ENVIRONMENT CONFIGURATION
 
load_dotenv()
 
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST") or "eu-west.cloud.hopsworks.ai"
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT") or "aqi_project_10pearls"
HOPSWORKS_PORT = int(os.getenv("HOPSWORKS_PORT") or 443)
 
# Dashboard forecast target

TARGET_COL = "target_aqi_72h"
 
FEATURE_COLS = [
    "pm25",
    "pm10",
    "o3",
    "no2",
    "so2",
    "co",
    "temp",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_deg",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "aqi_change_rate",
    "aqi_rolling_6h",
    "aqi_rolling_24h",
]
 
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
 
# LOAD DATA
 
def load_features() -> pd.DataFrame:
    """Load AQI features from Hopsworks Feature Store."""
 
    if not HOPSWORKS_KEY:
        raise ValueError("Missing HOPSWORKS_API_KEY. Please set it in your .env or GitHub Secrets.")
 
    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        port=HOPSWORKS_PORT,
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_KEY,
    )
 
    fs = project.get_feature_store()
 
    fg = fs.get_feature_group(
        name="aqi_features",
        version=1,
    )
 
    df = fg.read()
 
    print(f"[OK] Loaded {len(df)} rows from Hopsworks Feature Store")
    print(f"[OK] Columns: {list(df.columns)}")
 
    return df
 
# PREPARE DATA
 
def prepare_data(df: pd.DataFrame):
    """
    Prepare training and testing data. 
    Live rows with missing future targets are removed before training.
    """
 
    df = df.sort_values("timestamp")
 
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
 
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
    )
 
    print(f"[OK] Training data shape: {X_train.shape}")
    print(f"[OK] Testing data shape: {X_test.shape}")
 
    return X_train, X_test, y_train, y_test
 
# EVALUATION
 
def evaluate(name: str, y_true, y_pred) -> dict:
    """Evaluate model performance using RMSE, MAE, and R²."""
 
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
 
    print(f"[{name}] RMSE={rmse:.2f} MAE={mae:.2f} R²={r2:.4f}")
 
    return {
        "model": name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
 
# SKLEARN MODELS
 
def train_sklearn_models(X_train, X_test, y_train, y_test):
    """Train RandomForest, GradientBoost, and Ridge models."""
 
    results = []
    trained_models = {}
 
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            n_jobs=-1,
            random_state=42,
        ),
        "GradientBoost": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
        ),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]),
    }
 
    for name, model in models.items():
        print(f"\nTraining {name}...")
 
        model.fit(X_train, y_train)
 
        preds = model.predict(X_test)
 
        metrics = evaluate(name, y_test, preds)
        results.append(metrics)
        trained_models[name] = model
 
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)
 
        print(f"[OK] Saved {name} model -> {model_path}")
 
    return results, trained_models
 
# LSTM MODEL
 
def train_lstm(X_train, X_test, y_train, y_test, seq_len=24):
    """
    Training LSTM as an experimental comparison model.
 
    LSTM is not used as the final production model because the Streamlit dashboard
    uses sklearn .pkl artifacts and SHAP/LIME are more stable with sklearn models.
    """
 
    print("\nTraining LSTM experimental model...")
 
    scaler = StandardScaler()
 
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    def make_sequences(X, y, seq):
        xs = []
        ys = []
 
        for i in range(seq, len(X)):
            xs.append(X[i - seq:i])
            ys.append(y[i])
 
        return np.array(xs), np.array(ys)
 
    if len(X_train_scaled) < seq_len + 10 or len(X_test_scaled) < seq_len + 1:
        print("[WARNING] Not enough data for LSTM sequences. Skipping LSTM.")
        return None, None
 
    X_train_seq, y_train_seq = make_sequences(X_train_scaled, y_train, seq_len)
    X_test_seq, y_test_seq = make_sequences(X_test_scaled, y_test, seq_len)
 
    model = Sequential([
        LSTM(
            64,
            input_shape=(seq_len, X_train.shape[1]),
            return_sequences=True,
        ),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
 
    model.compile(
        optimizer="adam",
        loss="mse",
    )
 
    early_stop = EarlyStopping(
        patience=5,
        restore_best_weights=True,
    )
    model.fit(
        X_train_seq,
        y_train_seq,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )
 
    preds = model.predict(X_test_seq).flatten()
 
    metrics = evaluate("LSTM", y_test_seq, preds)
 
    lstm_model_path = os.path.join(MODELS_DIR, "lstm_model.h5")
    lstm_scaler_path = os.path.join(MODELS_DIR, "lstm_scaler.pkl")
 
    model.save(lstm_model_path)
    joblib.dump(scaler, lstm_scaler_path)
 
    print(f"[OK] Saved LSTM model -> {lstm_model_path}")
    print(f"[OK] Saved LSTM scaler -> {lstm_scaler_path}")
 
    return metrics, model
 
# SHAP EXPLAINABILITY
 
def explain_model_shap(model, X_test, feature_names):
    """
    Generate SHAP explanations for the production model. 
    SHAP explains which features influence AQI predictions.
    """
 
    print("\nComputing SHAP explanations...")
 
    try:
        X_sample = pd.DataFrame(
            X_test[:200],
            columns=feature_names,
        )
 
        X_single = X_sample.iloc[[0]]
 
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
 
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
 
            expected_value = explainer.expected_value
 
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[0]
 
        except Exception:
            explainer = shap.Explainer(model.predict, X_sample)
            shap_exp = explainer(X_sample)
            shap_values = shap_exp.values
            expected_value = shap_exp.base_values[0]
 
        plt.figure(figsize=(10, 6))
 
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
        )
 
        shap_summary_path = os.path.join(MODELS_DIR, "shap_summary.png")
        plt.savefig(
            shap_summary_path,
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()
 
        print(f"[OK] SHAP summary plot saved -> {shap_summary_path}")
 
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=expected_value,
                data=X_single.values[0],
                feature_names=feature_names,
            ),
            show=False,
        )
 
        shap_waterfall_path = os.path.join(MODELS_DIR, "shap_waterfall_sample.png")
        plt.savefig(
            shap_waterfall_path,
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()
 
        print(f"[OK] SHAP waterfall plot saved -> {shap_waterfall_path}")
 
    except Exception as e:
        print(f"[WARNING] SHAP explanation skipped: {e}")
 
# LIME EXPLAINABILITY
 
def explain_model_lime(model, X_train, X_test, feature_names):
    """
    Generate LIME explanation for one sample prediction.
 
    LIME provides local feature importance for a single prediction.
    """
 
    print("\nRunning LIME explanation...")
 
    try:
        X_train_df = pd.DataFrame(
            X_train,
            columns=feature_names,
        )
 
        X_test_df = pd.DataFrame(
            X_test,
            columns=feature_names,
        )
 
        explainer = LimeTabularExplainer(
            training_data=X_train_df.values,
            feature_names=feature_names,
            mode="regression",
            random_state=42,
        )
 
        sample = X_test_df.iloc[0].values
 
        explanation = explainer.explain_instance(
            data_row=sample,
            predict_fn=model.predict,
            num_features=10,
        )
 
        lime_html_path = os.path.join(MODELS_DIR, "lime_explanation.html")
        lime_png_path = os.path.join(MODELS_DIR, "lime_explanation.png")
 
        explanation.save_to_file(lime_html_path)
 
        fig = explanation.as_pyplot_figure()
        fig.savefig(
            lime_png_path,
            bbox_inches="tight",
            dpi=150,
        )
        plt.close(fig)
 
        print(f"[OK] LIME HTML saved -> {lime_html_path}")
        print(f"[OK] LIME PNG saved -> {lime_png_path}")
 
        return explanation
 
    except Exception as e:
        print(f"[WARNING] LIME explanation skipped: {e}")
        return None
 
# SAVE METRICS
 
def save_metrics(results):
    """Save metrics to models/metrics.json."""
 
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
 
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
 
    print(f"[OK] Metrics saved -> {metrics_path}")
 
# MODEL REGISTRY
 
def register_best_model(results: list, trained_models: dict):
    """
    FIX: Register the BEST model (lowest RMSE) instead of always GradientBoost.
    This ensures the most accurate model goes to production automatically.
    """
 
    if not HOPSWORKS_KEY:
        raise ValueError("Missing HOPSWORKS_API_KEY. Cannot register model.")
 
    sklearn_results = [r for r in results if r["model"] in trained_models]
    best = min(sklearn_results, key=lambda x: x["rmse"])
 
    print(f"\n[OK] Best model selected: {best['model']} (RMSE={best['rmse']:.2f}, R²={best['r2']:.4f})")
 
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(trained_models[best["model"]], best_model_path)
    print(f"[OK] Best model saved -> {best_model_path}")
 
    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        port=HOPSWORKS_PORT,
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_KEY,
    )
 
    model_registry = project.get_model_registry()
 
    hw_model = model_registry.python.create_model(
        name="aqi_predictor",
        metrics={
            "rmse": best["rmse"],
            "mae": best["mae"],
            "r2": best["r2"],
        },
        description=f"Best model: {best['model']} | target: {TARGET_COL}",
    )
 
    hw_model.save(MODELS_DIR)
 
    print("[OK] Model artifacts registered in Hopsworks Model Registry")
 
    return best
 
# MAIN PIPELINE
 
def run():
    """Run complete training pipeline."""
 
    print("=== Training Pipeline Started ===")
 
    df = load_features()
 
    X_train, X_test, y_train, y_test = prepare_data(df)
 
    all_results = []
 
    sklearn_results, sklearn_models = train_sklearn_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )
 
    all_results.extend(sklearn_results)
 
    lstm_metrics, _ = train_lstm(
        X_train,
        X_test,
        y_train,
        y_test,
    )
 
    if lstm_metrics:
        all_results.append(lstm_metrics)
 
    best = register_best_model(all_results, sklearn_models)
    production_model = sklearn_models[best["model"]]
 
    print(f"\nGenerating explanations for production model: {best['model']}")
 
    explain_model_shap(
        production_model,
        X_test,
        FEATURE_COLS,
    )
 
    explain_model_lime(
        production_model,
        X_train,
        X_test,
        FEATURE_COLS,
    )
 
    save_metrics(all_results)
 
    print("=== Training Pipeline Complete ===")
    print(f"\n{'='*40}")
    print(f"Production model : {best['model']}")
    print(f"RMSE             : {best['rmse']:.2f}")
    print(f"MAE              : {best['mae']:.2f}")
    print(f"R²               : {best['r2']:.4f}")
    print(f"{'='*40}")
 
 
if __name__ == "__main__":
    run()
