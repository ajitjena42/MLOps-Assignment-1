"""HeartGuard — An Observable ML Pipeline: Production FastAPI Serving Application.

Features:
- Validated Pydantic v2 ingestion with strict clinical boundary checking
- Modernized async lifespan lifecycle management with zero-IO in-memory serving
- Sub-millisecond latency tracking and real-time and batch scoring
- Statistical feature and concept drift monitoring engine (/drift)
- Native Prometheus metrics (/metrics) for request counts, latency histograms, risk summaries, and drift gauges
- Structured JSON access logging with request correlation IDs
- Production health and readiness probes (/health)
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

from src.api.middleware import StructuredLoggingMiddleware
from src.api.schemas import (
    BatchPatientData,
    BatchPredictionResponse,
    DriftEvaluationRequest,
    DriftReportResponse,
    HealthResponse,
    PatientData,
    PredictionResponse,
)
from src.config import settings
from src.models.drift import compute_drift, load_baseline_distribution
from src.models.predict import _MODEL, load_model, predict_batch, predict_single

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")

# ==================== PROMETHEUS METRICS ====================
HTTP_REQUESTS_TOTAL = Counter(
    "heart_disease_api_requests_total", "Total count of HTTP requests received", ["method", "endpoint", "status_code"]
)
PREDICTIONS_TOTAL = Counter(
    "heart_disease_predictions_total",
    "Total predictions broken down by clinical risk classification",
    ["prediction_class", "diagnosis_label"],
)
INFERENCE_LATENCY_SECONDS = Histogram(
    "heart_disease_inference_latency_seconds",
    "Time spent executing model pipeline inference",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
DISEASE_PROBABILITY_SUMMARY = Summary(
    "heart_disease_risk_probability", "Estimated probability of positive disease diagnosis"
)
MODEL_STATUS_GAUGE = Gauge(
    "heart_disease_model_loaded",
    "Status indicator showing whether machine learning pipeline is loaded (1 = Loaded, 0 = Failed)",
)
DRIFT_DETECTED_GAUGE = Gauge(
    "heart_disease_drift_detected",
    "Binary indicator showing whether statistical feature drift is detected (1 = Drift Detected, 0 = In Baseline)",
)
DRIFT_RATIO_GAUGE = Gauge(
    "heart_disease_drift_ratio", "Proportion of evaluated features exhibiting statistically significant drift"
)

# Application Lifecycle Counters
START_TIME = time.time()
PREDICTIONS_COUNTER = 0


# ==================== APPLICATION LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle cleanly."""
    logger.info("Initializing HeartGuard API with async lifespan management...")

    # 1. Load trained ML pipeline
    try:
        pipeline = load_model(settings.get_model_path())
        app.state.pipeline = pipeline
        MODEL_STATUS_GAUGE.set(1)
        logger.info(f"Model pipeline successfully loaded into memory from {settings.get_model_path()}")
    except Exception as e:
        app.state.pipeline = None
        MODEL_STATUS_GAUGE.set(0)
        logger.error(f"Failed to load model pipeline during startup: {e}")

    # 2. Load baseline distribution profile for drift detection
    try:
        baseline_stats = load_baseline_distribution(settings.BASELINE_STATS_PATH)
        app.state.baseline_stats = baseline_stats
        DRIFT_DETECTED_GAUGE.set(0)
        DRIFT_RATIO_GAUGE.set(0.0)
        logger.info("Baseline statistical distribution profile loaded successfully.")
    except Exception as e:
        app.state.baseline_stats = None
        logger.warning(f"Could not load baseline statistics profile: {e}")

    yield

    logger.info("Shutting down HeartGuard API.")


# ==================== FASTAPI APP DEFINITION ====================
app = FastAPI(
    title="HeartGuard — An Observable ML Pipeline",
    description=(
        "Production-grade, observable MLOps model serving API with integrated feature transformation, "
        "inference latency tracking, statistical drift monitoring, and Prometheus telemetry."
    ),
    version=settings.MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Attach custom structured logging middleware
app.add_middleware(StructuredLoggingMiddleware)


# ==================== ENDPOINTS ====================
@app.get("/app", response_class=HTMLResponse, tags=["Web Application"])
def web_app():
    """Interactive diagnostic web application for clinical risk assessment."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>HeartGuard Diagnostic Platform</h1><p>Template not found.</p>")


@app.get("/", tags=["General"])
def root(request: Request):
    """Root endpoint providing service metadata or redirecting to web app if accessed via browser."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept:
        return web_app()

    return {
        "service": settings.PROJECT_NAME,
        "student_id": settings.STUDENT_ID,
        "institution": settings.INSTITUTION,
        "version": settings.MODEL_VERSION,
        "status": "online",
        "web_app": "/app",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "drift": "/drift",
        "endpoints": {
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "drift_evaluation": "POST /drift",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health(request: Request):
    """Liveness and readiness probe for Kubernetes and Docker health checks.

    Checks in-memory application state with fallback to cached loader.
    """
    pipeline = getattr(request.app.state, "pipeline", None) or _MODEL
    if pipeline is None:
        try:
            pipeline = load_model(settings.get_model_path())
            request.app.state.pipeline = pipeline
        except Exception:
            pipeline = None

    model_loaded = pipeline is not None
    status_str = "healthy" if model_loaded else "degraded"

    uptime = round(time.time() - START_TIME, 2)
    return HealthResponse(
        status=status_str,
        model_loaded=model_loaded,
        version=settings.MODEL_VERSION,
        environment=settings.ENVIRONMENT,
        uptime_seconds=uptime,
        total_predictions_served=PREDICTIONS_COUNTER,
    )


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics scrape target."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(patient: PatientData, request: Request):
    """Predict heart disease presence/risk for an individual patient.

    Accepts 13 clinical biomarkers, transforms them through the scikit-learn
    pipeline, and returns binary prediction with calibrated risk confidence.
    """
    global PREDICTIONS_COUNTER
    start_time = time.perf_counter()
    pipeline = getattr(request.app.state, "pipeline", None)

    try:
        patient_dict = patient.model_dump()
        result = predict_single(patient_dict, pipeline=pipeline)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        latency_sec = latency_ms / 1000.0

        # Prometheus metrics recording
        PREDICTIONS_COUNTER += 1
        PREDICTIONS_TOTAL.labels(prediction_class=str(result["prediction"]), diagnosis_label=result["label"]).inc()
        INFERENCE_LATENCY_SECONDS.observe(latency_sec)
        DISEASE_PROBABILITY_SUMMARY.observe(result["probability_disease"])
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status_code="200").inc()

        return PredictionResponse(
            prediction=result["prediction"],
            label=result["label"],
            probability_disease=result["probability_disease"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            latency_ms=latency_ms,
            model_version=settings.MODEL_VERSION,
        )
    except Exception as e:
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status_code="500").inc()
        req_id = getattr(request.state, "request_id", "unknown")
        logger.error(f"Inference error during /predict [req_id={req_id}]: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failure. Verify inputs or check server logs.",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def batch_predict(batch: BatchPatientData, request: Request):
    """Predict cardiovascular risk for a batch of multiple patient records."""
    global PREDICTIONS_COUNTER
    start_time = time.perf_counter()
    pipeline = getattr(request.app.state, "pipeline", None)

    try:
        records = [p.model_dump() for p in batch.patients]
        results = predict_batch(records, pipeline=pipeline)

        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        avg_latency_ms = round(total_latency_ms / len(records), 2)

        formatted_predictions = []
        for r in results:
            PREDICTIONS_COUNTER += 1
            PREDICTIONS_TOTAL.labels(prediction_class=str(r["prediction"]), diagnosis_label=r["label"]).inc()
            DISEASE_PROBABILITY_SUMMARY.observe(r["probability_disease"])

            formatted_predictions.append(
                PredictionResponse(
                    prediction=r["prediction"],
                    label=r["label"],
                    probability_disease=r["probability_disease"],
                    confidence=r["confidence"],
                    probabilities=r["probabilities"],
                    latency_ms=avg_latency_ms,
                    model_version=settings.MODEL_VERSION,
                )
            )

        INFERENCE_LATENCY_SECONDS.observe(total_latency_ms / 1000.0)
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict/batch", status_code="200").inc()

        return BatchPredictionResponse(
            predictions=formatted_predictions, total_records=len(records), total_latency_ms=total_latency_ms
        )
    except Exception as e:
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict/batch", status_code="500").inc()
        req_id = getattr(request.state, "request_id", "unknown")
        logger.error(f"Error during /predict/batch [req_id={req_id}]: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch inference execution failure. Verify inputs or check server logs.",
        )


@app.get("/drift", tags=["Monitoring"])
def get_drift_status(request: Request):
    """Retrieve current statistical baseline distribution profile and drift monitoring metadata."""
    baseline = getattr(request.app.state, "baseline_stats", None)
    if not baseline:
        baseline = load_baseline_distribution()

    return {
        "status": "active",
        "baseline_reference": baseline.get("dataset_name", "UCI Cleveland"),
        "total_baseline_records": baseline.get("total_records", 0),
        "target_prevalence_baseline": baseline.get("target_distribution", {}),
        "monitored_features": list(baseline.get("features", {}).keys()),
        "thresholds": {
            "numerical_ks_pvalue": settings.DRIFT_PVALUE_THRESHOLD,
            "categorical_psi": settings.DRIFT_PSI_THRESHOLD,
        },
    }


@app.post("/drift", response_model=DriftReportResponse, tags=["Monitoring"])
def evaluate_drift(payload: DriftEvaluationRequest, request: Request):
    """Evaluate statistical feature drift for a batch of patient inference records against baseline."""
    baseline = getattr(request.app.state, "baseline_stats", None)
    try:
        records = [p.model_dump() for p in payload.records]
        report = compute_drift(records, baseline_stats=baseline)

        # Update Prometheus Drift Gauges
        DRIFT_DETECTED_GAUGE.set(1 if report["drift_detected"] else 0)
        DRIFT_RATIO_GAUGE.set(report["drift_ratio"])
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/drift", status_code="200").inc()

        return DriftReportResponse(
            drift_detected=report["drift_detected"],
            sample_size=report["sample_size"],
            total_features_evaluated=report["total_features_evaluated"],
            drifted_features_count=report["drifted_features_count"],
            drifted_features=report["drifted_features"],
            drift_ratio=report["drift_ratio"],
            baseline_reference=report["baseline_reference"],
            feature_metrics=report["feature_metrics"],
        )
    except Exception as e:
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/drift", status_code="500").inc()
        req_id = getattr(request.state, "request_id", "unknown")
        logger.error(f"Error during /drift evaluation [req_id={req_id}]: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Statistical drift evaluation failure. Verify payload or check server logs.",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host=settings.HOST, port=settings.PORT, reload=True)
