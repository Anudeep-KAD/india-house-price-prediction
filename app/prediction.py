import os
import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "random_forest_model.pkl")
PREP_PATH  = os.path.join(ROOT, "models", "preprocessor.pkl")

NUMERICAL_FEATURES = [
    "BHK", "Size_in_SqFt", "Year_Built", "Floor_No", "Total_Floors",
    "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Accessibility", "Parking_Space", "Security", "Amenities",
]
CATEGORICAL_FEATURES = [
    "State", "City", "Property_Type", "Furnished_Status",
    "Facing", "Owner_Type", "Availability_Status",
]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

_model = None
_preprocessor = None


def load_model():
    global _model, _preprocessor
    _model = joblib.load(MODEL_PATH)
    _preprocessor = joblib.load(PREP_PATH)


def is_model_loaded() -> bool:
    return _model is not None and _preprocessor is not None


def predict(features: dict) -> float:
    if not is_model_loaded():
        load_model()
    df = pd.DataFrame([features])[ALL_FEATURES]
    X = _preprocessor.transform(df)
    price = float(_model.predict(X)[0])
    return round(price, 2)
