# VIVA QUESTIONS & ANSWERS
## India House Price MLOps Project

### 1. What is MLOps?
MLOps combines ML development with DevOps. It covers training, testing, packaging, deploying, monitoring, and retraining ML models. Goal: make ML systems reliable, reproducible, and scalable.

### 2. What problem does this project solve?
Converts a house price prediction ML model into a production system with API serving, Docker packaging, Kubernetes orchestration, Prometheus monitoring, drift detection, and CI/CD automation.

### 3. What is FastAPI and why use it?
FastAPI is a Python web framework for REST APIs. Used because it is fast, auto-generates Swagger docs, validates inputs with Pydantic, and is the industry standard for Python ML APIs.

### 4. What is a REST API?
REST API is a way for software to communicate over HTTP. Clients send GET/POST requests to endpoints, server returns JSON. Here, POST /predict receives house features and returns a price.

### 5. What does the /health endpoint do?
Checks whether the API is running and the model is loaded. Returns 200 if healthy, 503 if not. Kubernetes uses it as a liveness and readiness probe.

### 6. What is Docker and why use it?
Docker packages an application plus all dependencies into a container. Guarantees it runs identically everywhere — laptop, CI server, cloud. Eliminates "works on my machine" problems.

### 7. Docker vs Virtual Machine?
VM virtualises the full OS (slow, GBs). Docker containers share the host OS kernel but isolate the app (fast, MBs). Docker starts in seconds; VMs take minutes.

### 8. What is Kubernetes?
Kubernetes (K8s) automatically manages deploying, scaling, and restarting containers. If a container crashes, it restarts it. If traffic increases, it starts more containers.

### 9. What is a Pod in Kubernetes?
The smallest deployable unit. A Pod wraps one or more containers sharing the same network. Each Pod here runs one instance of the house-price-api container.

### 10. What is a Deployment in Kubernetes?
A Deployment manages a set of identical Pods. You specify "run 2 replicas" and it ensures exactly 2 are always running. If one crashes, it automatically creates a replacement.

### 11. What is a Service in Kubernetes?
Provides a stable network address to access Pods. Pods have dynamic IPs; a Service acts as a fixed endpoint (load balancer) routing traffic to healthy Pods.

### 12. What are replicas and why have 2?
Replicas are copies of the same Pod running simultaneously. 2 replicas means: if one crashes the other keeps serving (high availability), load is shared (better performance), and rolling updates have zero downtime.

### 13. What happens if a Pod crashes?
Kubernetes detects failure via liveness probe, deletes the failed Pod, starts a new one. The Service routes traffic to the remaining healthy Pod. Self-healing, no human intervention needed.

### 14. What is a readinessProbe?
Tells Kubernetes when a Pod is ready to accept traffic. Calls GET /health. Until the model is loaded and /health returns 200, Kubernetes does NOT send traffic to that Pod.

### 15. What is a livenessProbe?
Tells Kubernetes whether a Pod is still alive. Calls GET /health. If it fails 3 times, Kubernetes restarts the Pod. Recovers from deadlocks or out-of-memory situations.

### 16. What is HorizontalPodAutoscaler (HPA)?
Automatically scales the number of Pods based on CPU/memory usage. This project scales between 2 and 8 replicas when CPU exceeds 70%. Adds Pods under high traffic, removes them when traffic drops.

### 17. What is Prometheus?
Open-source monitoring system. It periodically scrapes metrics from the /metrics endpoint and stores them as time-series data. Also supports alert rules that trigger when thresholds are exceeded.

### 18. What metrics does Prometheus collect here?
- http_requests_total — requests by endpoint and status code
- http_request_duration_seconds — latency histogram
- predictions_total — prediction call count
- prediction_duration_seconds — inference time
- http_errors_total — error count
- model_loaded — whether model is loaded (0 or 1)

### 19. What is Grafana?
Data visualisation tool that connects to Prometheus and displays metrics as dashboards with graphs and gauges. Makes it easy to visually monitor traffic, latency, and errors.

### 20. What alert rules did you create?
1. APIDown — API unreachable >1 min (critical)
2. HighErrorRate — >5% 5xx errors (warning)
3. HighPredictionLatency — p95 latency >2s (warning)
4. ModelNotLoaded — model gauge = 0 (critical)

### 21. What is CI/CD?
CI = automatically test code on every push. CD = automatically build and deliver tested code. Together they prevent broken code reaching production and eliminate manual deployment steps.

### 22. What does your GitHub Actions pipeline do?
On push to main: checkout code → setup Python 3.12 → install dependencies → train model → run 20 pytest tests (stops if any fail) → build Docker image → push to Docker Hub.

### 23. What is data drift?
When statistical distribution of production inputs changes vs training data. Example: model trained on avg flat size 1000 sqft, but production data averages 1200 sqft. Model predictions become less accurate.

### 24. How did you detect data drift?
Numerical features: KS-test (compares cumulative distributions). Categorical features: Chi-squared test (compares frequency distributions). p-value < 0.05 = drift flagged.

### 25. What is concept drift?
When the relationship between features and target changes even if feature distributions look the same. Example: a 3BHK in Bengaluru costs 150 lakhs in 2020 but 200 lakhs in 2024. Requires monitoring actual prediction accuracy with real labels.

### 26. Does data drift mean the model is wrong?
No. Data drift is a warning signal. The model might still predict correctly. Concept drift (relationship change) is what actually degrades accuracy. Always check actual prediction errors before retraining.

### 27. What is RandomForestRegressor?
An ensemble model that trains 100 decision trees on random data subsets and averages their predictions. Handles non-linear relationships, robust to outliers, works well with mixed feature types.

### 28. What is MAE?
Mean Absolute Error — average absolute difference between predicted and actual prices. This project: MAE = 6.75 lakhs. Predictions are off by ~6.75 lakhs on average. In the same units as target.

### 29. What is RMSE?
Root Mean Squared Error — square root of average squared errors. This project: RMSE = 9.89 lakhs. Penalises large errors more than MAE. Useful when big mistakes are especially costly.

### 30. What is R² score?
Coefficient of determination — how much price variance the model explains. This project: R² = 0.973 (97.3% of variance explained). Range 0–1; closer to 1 is better. Above 0.9 is excellent.

### 31. Why use a ColumnTransformer/preprocessor?
Raw data has numerical features (BHK, size) and categorical features (State, City). ML models need all numbers. ColumnTransformer applies StandardScaler to numerical and OneHotEncoder to categorical features in one step.

### 32. Why save the preprocessor separately from the model?
At prediction time, incoming data must go through the exact same transformations as training data — same feature order, same encoding. Without saving the preprocessor, consistent predictions are impossible.

### 33. What is Pydantic?
Python data validation library. The /predict request body is a Pydantic model. FastAPI uses it to automatically validate all 19 fields — types, ranges, allowed values. Invalid requests get a 422 error automatically.

### 34. Docker image vs Docker container?
Image = static read-only snapshot (like an ISO file). Container = running instance of an image (like a running program). Many containers can run from the same image simultaneously.

### 35. What happens to Pod data when it restarts?
By default Pod storage is ephemeral — lost on restart. The model is fine because it is baked into the Docker image. For databases or logs, Kubernetes PersistentVolumes are needed.
