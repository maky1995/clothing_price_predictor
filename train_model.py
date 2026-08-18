"""
Trains the Random Forest price model and saves both the model and the
OneHotEncoder (needed to transform new inputs at prediction time).

Run this manually if you want to retrain from a fresh dataset:
    python train_model.py
Otherwise app.py will train automatically on first run.
"""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from data_utils import FEATURES, load_dataset, prepare_data

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "Price_Quota_Model.pkl"
ENCODER_FILE = BASE_DIR / "price_encoder.pkl"


def main():
    df = prepare_data(load_dataset())

    X = df[FEATURES]
    y = df["initial_price"]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_encoded = encoder.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_encoded, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)

    print(f"Saved model to: {MODEL_FILE}")
    print(f"Saved encoder to: {ENCODER_FILE}")
    print(f"Training rows: {len(df)}")
    print(f"Features: {FEATURES}")


if __name__ == "__main__":
    main()
