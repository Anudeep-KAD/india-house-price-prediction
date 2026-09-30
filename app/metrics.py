from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total prediction requests",
)

PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Time taken to generate a prediction",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)

ERROR_COUNT = Counter(
    "http_errors_total",
    "Total HTTP errors",
    ["endpoint"],
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the ML model is currently loaded (1=yes, 0=no)",
)
