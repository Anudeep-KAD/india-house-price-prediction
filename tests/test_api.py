import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "State": "Karnataka",
    "City": "Bengaluru",
    "Property_Type": "Apartment",
    "BHK": 3,
    "Size_in_SqFt": 1200,
    "Year_Built": 2015,
    "Furnished_Status": "Semi-Furnished",
    "Floor_No": 5,
    "Total_Floors": 12,
    "Age_of_Property": 9,
    "Nearby_Schools": 3,
    "Nearby_Hospitals": 2,
    "Public_Transport_Accessibility": 7,
    "Parking_Space": 1,
    "Security": 1,
    "Amenities": 5,
    "Facing": "North-East",
    "Owner_Type": "Builder",
    "Availability_Status": "Ready to Move",
}


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "India House Price Prediction API"
    assert data["status"] == "running"
    assert "version" in data


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_valid():
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert "predicted_price_lakhs" in data
    price = data["predicted_price_lakhs"]
    assert isinstance(price, float)
    assert price > 0


def test_predict_price_range():
    r = client.post("/predict", json=VALID_PAYLOAD)
    price = r.json()["predicted_price_lakhs"]
    # Reasonable range for a 3BHK 1200sqft in Bengaluru
    assert 20 < price < 2000, f"Price out of reasonable range: {price}"


def test_predict_missing_field():
    payload = dict(VALID_PAYLOAD)
    del payload["BHK"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_state():
    payload = dict(VALID_PAYLOAD)
    payload["State"] = "InvalidState"
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_bhk_out_of_range():
    payload = dict(VALID_PAYLOAD)
    payload["BHK"] = 99
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"http_requests_total" in r.content


def test_predict_mumbai():
    payload = dict(VALID_PAYLOAD)
    payload["State"] = "Maharashtra"
    payload["City"] = "Mumbai"
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    price = r.json()["predicted_price_lakhs"]
    assert price > 0
