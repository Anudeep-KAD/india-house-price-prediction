"""
training/train.py
Train a RandomForestRegressor on the India House Price dataset.
Saves model, preprocessor, and metrics.json.
"""
import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(ROOT, "data", "house_prices.csv")
MODEL_DIR  = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
PREP_PATH  = os.path.join(MODEL_DIR, "preprocessor.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Feature definitions ────────────────────────────────────────────────────────
TARGET = "Price_in_Lakhs"
DROP_COLS = ["ID", "Locality", TARGET]

CATEGORICAL_FEATURES = [
    "State", "City", "Property_Type", "Furnished_Status",
    "Facing", "Owner_Type", "Availability_Status",
]
NUMERICAL_FEATURES = [
    "BHK", "Size_in_SqFt", "Year_Built", "Floor_No", "Total_Floors",
    "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Accessibility", "Parking_Space", "Security", "Amenities",
]


def load_data():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Target range: {df[TARGET].min():.2f} – {df[TARGET].max():.2f} lakhs")
    return df


def build_preprocessor():
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def train():
    t0 = time.time()
    df = load_data()

    X = df.drop(columns=DROP_COLS)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    preprocessor = build_preprocessor()

    print("Fitting preprocessor...")
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    print("Training RandomForestRegressor (n_estimators=100, n_jobs=-1)...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train_t, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test_t)
    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))

    print(f"\n  MAE  = {mae:.4f} lakhs")
    print(f"  RMSE = {rmse:.4f} lakhs")
    print(f"  R²   = {r2:.4f}")

    import sklearn, joblib as jb, pandas as pdv
    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": X_train_t.shape[1],
        "training_time_sec": round(time.time() - t0, 2),
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "pandas_version": pdv.__version__,
        "model": "RandomForestRegressor",
        "n_estimators": 100,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }

    print(f"\nSaving model  → {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    print(f"Saving preprocessor → {PREP_PATH}")
    joblib.dump(preprocessor, PREP_PATH)
    print(f"Saving metrics → {METRICS_PATH}")
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete in {metrics['training_time_sec']}s ✓")
    return metrics


if __name__ == "__main__":
    train()
