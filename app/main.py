"""Heart disease risk API: serves a saved scikit-learn Pipeline (preprocessing + model)."""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/model.joblib"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart-api")

state = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from %s", MODEL_PATH)
    state["model"] = joblib.load(MODEL_PATH)
    logger.info("Model loaded")
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)  # adds GET /metrics for Prometheus


class PatientData(BaseModel):
    # Column names must match what your training pipeline was fitted on.
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=4)
    trestbps: float = Field(..., gt=0)
    chol: float = Field(..., gt=0)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., gt=0)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0)
    slope: int = Field(..., ge=0, le=3)
    ca: float = Field(..., ge=0, le=4)
    thal: float = Field(..., ge=0, le=7)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, response.status_code, ms)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.post("/predict")
def predict(patient: PatientData):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    row = pd.DataFrame([patient.model_dump()])
    proba_disease = float(model.predict_proba(row)[0][1])
    prediction = int(proba_disease >= 0.5)
    confidence = proba_disease if prediction == 1 else 1 - proba_disease
    logger.info("prediction=%d confidence=%.3f", prediction, confidence)
    return {
        "prediction": prediction,
        "label": "heart disease likely" if prediction else "heart disease unlikely",
        "confidence": round(confidence, 4),
        "probability_disease": round(proba_disease, 4),
    }
