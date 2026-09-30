# Production ML Model Deployment and Monitoring System
### India House Price Prediction — MLOps Project

---

## Problem Statement

Real estate pricing in India is complex and highly location-dependent. This project converts a machine learning model that predicts Indian residential property prices (in Lakhs ₹) into a **production-grade MLOps system** — covering model serving, containerisation, orchestration, monitoring, drift detection, and CI/CD automation.

---

## Objectives

1. Serve an ML model as a production REST API (FastAPI)
2. Containerise with Docker for consistent deployments
3. Orchestrate with Kubernetes for scalability and resilience
4. Monitor with Prometheus + Grafana
5. Detect data drift in production inputs
6. Automate testing and Docker builds via GitHub Actions CI/CD

---

## Architecture

```
Dataset (250k rows, 22 columns)
         ↓
ML Model Training (RandomForestRegressor)
         ↓
Saved Model + Preprocessor (.pkl)
         ↓
FastAPI Model Serving  →  /predict, /health, /metrics
         ↓
Docker Container  →  python:3.12-slim
         ↓
Kubernetes (2 replicas, HPA, readiness/liveness probes)
         ↓
Prometheus (scrapes /metrics every 10s)
         ↓
Grafana Dashboard (requests, latency, errors, predictions)

GitHub → GitHub Actions CI/CD → Tests → Docker Build → Push
Production Data → Drift Detection → KS-test / Chi²-test → Report
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| ML | scikit-learn (RandomForestRegressor) |
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Metrics | prometheus-client |
| Containers | Docker (python:3.12-slim) |
| Orchestration | Kubernetes / Minikube |
| Monitoring | Prometheus v2.53 + Grafana v11.1 |
| Drift Detection | SciPy (KS-test, Chi²-test) |
| CI/CD | GitHub Actions |
| Testing | pytest + httpx |

---

## Project Structure

```
mlops-house-price/
├── data/
│   ├── generate_dataset.py     # Synthetic dataset generator
│   └── house_prices.csv        # 250,000 rows, 22 columns
├── models/
│   ├── random_forest_model.pkl # Trained model (187MB)
│   ├── preprocessor.pkl        # ColumnTransformer
│   ├── metrics.json            # Actual training metrics
│   └── drift_report.json       # Latest drift detection results
├── training/
│   ├── train.py                # Training pipeline
│   └── evaluate.py             # Evaluation on test set
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app with middleware
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── prediction.py           # Model loading + inference
│   └── metrics.py              # Prometheus counters/histograms
├── tests/
│   ├── test_api.py             # API endpoint tests
│   ├── test_health.py          # Health check tests
│   └── test_model.py           # Model loading + inference tests
├── drift/
│   └── drift_detection.py      # KS-test + Chi² drift detection
├── monitoring/
│   ├── prometheus.yml          # Prometheus scrape config
│   └── alerts.yml              # Alert rules (4 alerts)
├── k8s/
│   ├── deployment.yaml         # 2-replica Deployment
│   ├── service.yaml            # NodePort Service
│   ├── hpa.yaml                # HorizontalPodAutoscaler
│   ├── configmap.yaml          # Prometheus config
│   ├── prometheus-deployment.yaml
│   └── grafana-deployment.yaml
├── .github/workflows/
│   └── ci-cd.yml               # GitHub Actions pipeline
├── Dockerfile                  # python:3.12-slim
├── docker-compose.yml          # API + Prometheus + Grafana
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── DEMO_GUIDE.md
└── VIVA_QUESTIONS.md
```

---

## ML Model

**Algorithm:** RandomForestRegressor  
**Training Data:** 200,000 rows  
**Test Data:** 50,000 rows  
**Features:** 19 input features → 82 after one-hot encoding

### Actual Results (verified, not estimated)

| Metric | Value |
|--------|-------|
| MAE | **6.75 lakhs** |
| RMSE | **9.89 lakhs** |
| R² | **0.9730** |
| Training time | 173.67 seconds |

### Features Used

**Numerical (12):** BHK, Size_in_SqFt, Year_Built, Floor_No, Total_Floors, Age_of_Property, Nearby_Schools, Nearby_Hospitals, Public_Transport_Accessibility, Parking_Space, Security, Amenities

**Categorical (7):** State, City, Property_Type, Furnished_Status, Facing, Owner_Type, Availability_Status

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check + model status |
| POST | `/predict` | Predict house price |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI |

### Example: POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "Availability_Status": "Ready to Move"
  }'
```

Response:
```json
{"predicted_price_lakhs": 158.83, "model_version": "1.0.0"}
```

---

## Docker

```bash
# Build
docker build -t house-price-api .

# Run
docker run -p 8000:8000 house-price-api

# Full stack (API + Prometheus + Grafana)
docker-compose up -d
```

Ports: API=8000, Prometheus=9090, Grafana=3000 (admin/admin123)

---

## Kubernetes

```bash
minikube start

kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/prometheus-deployment.yaml
kubectl apply -f k8s/grafana-deployment.yaml

kubectl get pods
kubectl get services

# Access API
minikube service house-price-api

# Scale manually
kubectl scale deployment house-price-api --replicas=3
```

---

## Prometheus Metrics Tracked

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Requests by method/endpoint/status |
| `http_request_duration_seconds` | Histogram | Request latency |
| `predictions_total` | Counter | Total predictions |
| `prediction_duration_seconds` | Histogram | Inference latency |
| `http_errors_total` | Counter | Errors by endpoint |
| `model_loaded` | Gauge | 1 if model is loaded |

### Alert Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| APIDown | API unreachable >1min | Critical |
| HighErrorRate | >5% 5xx errors | Warning |
| HighPredictionLatency | p95 latency >2s | Warning |
| ModelNotLoaded | model_loaded=0 | Critical |

---

## Data Drift Detection

```bash
python3 drift/drift_detection.py
```

Uses **KS-test** (numerical) and **Chi²-test** (categorical). Compares reference data (training set) vs simulated production data.

**Actual results from last run (5/19 features drifted):**
- BHK ⚠ drifted
- Size_in_SqFt ⚠ drifted
- Year_Built ⚠ drifted
- Age_of_Property ⚠ drifted
- Furnished_Status ⚠ drifted

---

## Testing

```bash
cd mlops-house-price
python3 -m pytest tests/ -v
```

**Result: 20/20 tests passed**

---

## Setup Instructions

```bash
# Clone / enter project
cd mlops-house-price

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python3 data/generate_dataset.py

# Train model
python3 training/train.py

# Start API locally
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run tests
python3 -m pytest tests/ -v

# Drift detection
python3 drift/drift_detection.py

# Docker full stack
docker-compose up -d
```

---

## CI/CD (GitHub Actions)

On every push to `main`:
1. Set up Python 3.12
2. Install dependencies
3. Generate dataset + train model
4. Run pytest (fails pipeline if tests fail)
5. Build Docker image
6. Push to Docker Hub (if `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets are set)

---

## Limitations

- Model is trained on synthetic data (realistic distributions, not real MLS data)
- Docker image is ~2GB due to scikit-learn + model file (187MB)
- Kubernetes deployment tested with manifests only; Minikube required for local verification
- Grafana dashboard must be configured manually via UI (datasource: Prometheus at http://prometheus:9090)
- No persistent storage for model versioning (use MLflow or DVC in production)

## Future Improvements

- Integrate MLflow for experiment tracking
- Add DVC for dataset versioning
- Implement model versioning with A/B testing
- Add authentication to the API (OAuth2/JWT)
- Real production data pipeline with Kafka
- Automated retraining trigger when drift is detected
- Grafana dashboard as code (provisioning YAML)
