import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_model_file_exists():
    path = os.path.join(ROOT, "models", "random_forest_model.pkl")
    assert os.path.exists(path), f"Model not found at {path}"


def test_preprocessor_file_exists():
    path = os.path.join(ROOT, "models", "preprocessor.pkl")
    assert os.path.exists(path), f"Preprocessor not found at {path}"


def test_metrics_file_exists():
    path = os.path.join(ROOT, "models", "metrics.json")
    assert os.path.exists(path)


def test_metrics_values():
    path = os.path.join(ROOT, "models", "metrics.json")
    with open(path) as f:
        m = json.load(f)
    assert m["r2"] > 0.5, f"R2 too low: {m['r2']}"
    assert m["mae"] > 0
    assert m["rmse"] > 0


def test_model_loads():
    path = os.path.join(ROOT, "models", "random_forest_model.pkl")
    model = joblib.load(path)
    assert model is not None


def test_preprocessor_loads():
    path = os.path.join(ROOT, "models", "preprocessor.pkl")
    prep = joblib.load(path)
    assert prep is not None


def test_model_prediction_shape():
    from app.prediction import load_model, predict
    load_model()
    result = predict({
        "State": "Delhi", "City": "New Delhi", "Property_Type": "Apartment",
        "BHK": 2, "Size_in_SqFt": 900, "Year_Built": 2010,
        "Furnished_Status": "Unfurnished", "Floor_No": 3, "Total_Floors": 10,
        "Age_of_Property": 14, "Nearby_Schools": 2, "Nearby_Hospitals": 1,
        "Public_Transport_Accessibility": 8, "Parking_Space": 1, "Security": 1,
        "Amenities": 4, "Facing": "South", "Owner_Type": "Owner",
        "Availability_Status": "Ready to Move",
    })
    assert isinstance(result, float)
    assert result > 0
