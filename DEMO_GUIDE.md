# DEMO GUIDE — College Demonstration
## India House Price Prediction MLOps System

---

## Pre-Demo Checklist (do this 10 minutes before)

```bash
cd mlops-house-price
pip install -r requirements.txt    # ensure all packages present
python3 -m pytest tests/ -v        # confirm 20/20 pass
```

---

## DEMO STEP 1 — Start the API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Expected output:**
```
✓ Model loaded at startup
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## DEMO STEP 2 — Open Swagger UI

Open browser: **http://localhost:8000/docs**

Show:
- All 4 endpoints listed
- Request schema with all 19 fields
- Response schema

---

## DEMO STEP 3 — Send a prediction via Swagger

Click **POST /predict → Try it out → Execute**

Use this payload:
```json
{
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
}
```

**Expected response:** `{"predicted_price_lakhs": 158.83, "model_version": "1.0.0"}`

---

## DEMO STEP 4 — Show /health endpoint

```bash
curl http://localhost:8000/health
```
```json
{"status": "healthy", "model_loaded": true, "version": "1.0.0"}
```

---

## DEMO STEP 5 — Show /metrics (Prometheus format)

```bash
curl http://localhost:8000/metrics | head -30
```

Point out the metrics: `http_requests_total`, `predictions_total`, `prediction_duration_seconds`

---

## DEMO STEP 6 — Run Tests

Open new terminal:
```bash
cd mlops-house-price
python3 -m pytest tests/ -v
```

Show: **20 passed**

Explain each test file:
- `test_api.py` — endpoint tests
- `test_health.py` — health check tests
- `test_model.py` — model loading tests

---

## DEMO STEP 7 — Docker

```bash
# Build the image
docker build -t house-price-api .

# Show the image
docker images | grep house-price-api

# Run the container
docker run -d -p 8000:8000 --name house-price-demo house-price-api

# Wait 30 seconds then test
sleep 30 && curl http://localhost:8000/health

# Stop
docker stop house-price-demo
```

---

## DEMO STEP 8 — Docker Compose (Full Stack)

```bash
docker-compose up -d

# Show running containers
docker-compose ps
```

Services running:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

---

## DEMO STEP 9 — Prometheus

Open **http://localhost:9090**

1. Go to **Status → Targets** — show house-price-api is UP
2. Go to **Graph**, enter query: `http_requests_total`
3. Show: `predictions_total`
4. Show: `model_loaded`

---

## DEMO STEP 10 — Grafana Setup

Open **http://localhost:3000** (admin / admin123)

**First time only — add datasource:**
1. Left menu → **Connections → Data Sources**
2. Click **Add data source → Prometheus**
3. URL: `http://prometheus:9090`
4. Click **Save & Test** → should say "Data source is working"

**Create dashboard:**
1. Left menu → **Dashboards → New → New Dashboard**
2. Click **Add visualization**
3. Select **Prometheus** datasource
4. Add these panels one by one:

| Panel Title | PromQL Query |
|-------------|-------------|
| Total Requests | `sum(http_requests_total)` |
| Requests/sec | `sum(rate(http_requests_total[1m]))` |
| Prediction Count | `predictions_total` |
| Error Rate | `sum(rate(http_requests_total{status_code=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))` |
| Avg Latency | `rate(http_request_duration_seconds_sum[1m]) / rate(http_request_duration_seconds_count[1m])` |

---

## DEMO STEP 11 — Generate Traffic to See Metrics Change

Run this in a terminal to send 50 prediction requests:
```bash
for i in {1..50}; do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "State": "Maharashtra", "City": "Mumbai", "Property_Type": "Apartment",
      "BHK": 2, "Size_in_SqFt": 950, "Year_Built": 2018, "Furnished_Status": "Furnished",
      "Floor_No": 8, "Total_Floors": 20, "Age_of_Property": 6, "Nearby_Schools": 4,
      "Nearby_Hospitals": 3, "Public_Transport_Accessibility": 9, "Parking_Space": 1,
      "Security": 1, "Amenities": 7, "Facing": "West", "Owner_Type": "Builder",
      "Availability_Status": "Ready to Move"
    }' > /dev/null
  echo "Request $i sent"
done
```

Go to Grafana → watch metrics update in real time.

---

## DEMO STEP 12 — Data Drift Detection

Open a new terminal:
```bash
cd mlops-house-price
python3 drift/drift_detection.py
```

Expected output shows:
- 5 features with drift (BHK, Size_in_SqFt, Year_Built, Age_of_Property, Furnished_Status)
- 14 features with no drift
- Drift report saved to models/drift_report.json

**Explain:** "This simulates what happens when production data patterns change — for example, new flats being built larger, or buyer preferences shifting toward furnished apartments."

---

## DEMO STEP 13 — Kubernetes

```bash
minikube start

# Apply all manifests
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Watch pods start up
kubectl get pods -w

# Show 2 replicas running
kubectl get pods
kubectl get deployments
kubectl get services

# Access the API
minikube service house-price-api

# Scale to 3 replicas
kubectl scale deployment house-price-api --replicas=3
kubectl get pods  # show 3 pods
```

---

## DEMO STEP 14 — Show GitHub Actions

Go to your GitHub repository → **Actions tab**

Show:
- CI/CD pipeline YAML
- Three jobs: Test → Build → Push
- Status badges
- Artifacts (metrics.json)

**Explain:** "Every time I push code to main, GitHub automatically runs tests, and if all 20 pass, it builds the Docker image. This ensures we never deploy broken code."

---

## Key Numbers to Remember

| Item | Value |
|------|-------|
| Dataset size | 250,000 rows |
| Model | RandomForest (100 trees) |
| R² score | 0.973 |
| MAE | 6.75 lakhs |
| Tests | 20/20 pass |
| API port | 8000 |
| Prometheus port | 9090 |
| Grafana port | 3000 |
| Kubernetes replicas | 2 (auto-scales to 8) |
| Drift detected | 5/19 features |
