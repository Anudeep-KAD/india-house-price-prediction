import time
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.schemas import HouseFeatures, PredictionResponse
from app import prediction as pred_module
from app.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, PREDICTION_COUNT,
    PREDICTION_LATENCY, ERROR_COUNT, MODEL_LOADED,
)

VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pre-load the model
    try:
        pred_module.load_model()
        MODEL_LOADED.set(1)
        print("✓ Model loaded at startup")
    except Exception as e:
        MODEL_LOADED.set(0)
        print(f"✗ Model failed to load: {e}")
    yield
    # Shutdown
    print("API shutting down")


app = FastAPI(
    title="India House Price Prediction API",
    description="MLOps project — predicts house prices in lakhs for Indian properties.",
    version=VERSION,
    lifespan=lifespan,
)


# ── Middleware: track every request ───────────────────────────────────────────
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
    if response.status_code >= 400:
        ERROR_COUNT.labels(endpoint=endpoint).inc()
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "name": "India House Price Prediction API",
        "version": VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "metrics": "/metrics",
    }


@app.get("/health", tags=["Health"])
def health():
    if not pred_module.is_model_loaded():
        try:
            pred_module.load_model()
            MODEL_LOADED.set(1)
        except Exception as e:
            MODEL_LOADED.set(0)
            raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")
    MODEL_LOADED.set(1)
    return {"status": "healthy", "model_loaded": True, "version": VERSION}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: HouseFeatures):
    start = time.time()
    try:
        price = pred_module.predict(features.model_dump())
        PREDICTION_COUNT.inc()
        PREDICTION_LATENCY.observe(time.time() - start)
        return PredictionResponse(predicted_price_lakhs=price)
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
