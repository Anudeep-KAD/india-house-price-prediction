import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_status_code():
    r = client.get("/health")
    assert r.status_code == 200


def test_health_model_loaded():
    r = client.get("/health")
    assert r.json()["model_loaded"] is True


def test_health_has_version():
    r = client.get("/health")
    assert "version" in r.json()


def test_root_has_endpoints():
    r = client.get("/")
    data = r.json()
    for key in ("docs", "health", "predict", "metrics"):
        assert key in data
