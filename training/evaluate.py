"""
training/evaluate.py
Load saved model + preprocessor and re-evaluate on test data.
"""
import os, sys, json
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_PATH  = os.path.join(ROOT, "data", "house_prices.csv")
MODEL_PATH = os.path.join(ROOT, "models", "random_forest_model.pkl")
PREP_PATH  = os.path.join(ROOT, "models", "preprocessor.pkl")
METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")

TARGET = "Price_in_Lakhs"
DROP_COLS = ["ID", "Locality", TARGET]

def evaluate():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=DROP_COLS)
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Loading model and preprocessor...")
    model = joblib.load(MODEL_PATH)
    prep  = joblib.load(PREP_PATH)

    print("Running inference on test set...")
    X_t = prep.transform(X_test)
    y_pred = model.predict(X_t)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)

    print(f"\n{'='*40}")
    print("EVALUATION RESULTS")
    print(f"{'='*40}")
    print(f"  Test samples : {len(y_test):,}")
    print(f"  MAE          : {mae:.4f} lakhs")
    print(f"  RMSE         : {rmse:.4f} lakhs")
    print(f"  R²           : {r2:.4f}")
    print(f"{'='*40}")

    with open(METRICS_PATH) as f:
        saved = json.load(f)
    print(f"\nSaved metrics  →  MAE={saved['mae']}  RMSE={saved['rmse']}  R²={saved['r2']}")
    print("Metrics match ✓" if abs(r2 - saved["r2"]) < 0.01 else "⚠ Mismatch — model may have changed")

if __name__ == "__main__":
    evaluate()
