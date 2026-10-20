# HeartGuard MLOps Platform — Setup & Installation Guide
## Complete Step-by-Step Instructions for Local, Docker, and Kubernetes Environments

**Student ID:** `2025AE05839`  
**Course:** AIMLCZG523 — Machine Learning Operations (MLOps)  
**Institution:** Birla Institute of Technology & Science (BITS), Pilani  
**Assignment:** Experimental Learning Assignment 01 (Total Marks: 50)  
**System Name:** HeartGuard — Cardiovascular Risk Prediction & Monitoring Platform  

---

## Table of Contents
1. [Prerequisites & System Requirements](#1-prerequisites--system-requirements)
2. [Quick Verification (Under 2 Minutes)](#2-quick-verification-under-2-minutes)
3. [Local Python Environment Setup](#3-local-python-environment-setup)
   - [Option A: Python Virtual Environment (`venv` - Recommended)](#option-a-python-virtual-environment-venv---recommended)
   - [Option B: Conda Environment (`environment.yml`)](#option-b-conda-environment-environmentyml)
4. [Data Acquisition, Cleaning & Baseline Profiling](#4-data-acquisition-cleaning--baseline-profiling)
5. [Model Training & MLflow Experiment Tracking](#5-model-training--mlflow-experiment-tracking)
6. [Automated Testing & Code Quality Assurance](#6-automated-testing--code-quality-assurance)
7. [Local REST API Serving & Verification](#7-local-rest-api-serving--verification)
   - [Starting the FastAPI Microservice](#starting-the-fastapi-microservice)
   - [Interactive Web Interfaces](#interactive-web-interfaces)
   - [Testing Endpoints with cURL](#testing-endpoints-with-curl)
8. [Docker Containerization Setup](#8-docker-containerization-setup)
   - [Building the Image](#building-the-image)
   - [Running the Container](#running-the-container)
   - [Automated Docker Verification Script](#automated-docker-verification-script)
9. [Full-Stack Observability Stack (Docker Compose)](#9-full-stack-observability-stack-docker-compose)
10. [Production Kubernetes & Helm Deployment](#10-production-kubernetes--helm-deployment)
    - [Kubernetes Raw Manifests (Minikube / Cluster)](#kubernetes-raw-manifests-minikube--cluster)
    - [Helm Chart Deployment](#helm-chart-deployment)
11. [Troubleshooting & FAQs](#11-troubleshooting--faqs)
12. [Assignment Deliverable Mapping](#12-assignment-deliverable-mapping)

---

## 1. Prerequisites & System Requirements

Ensure the host machine meets the following baseline tool requirements:

| Tool | Recommended Version | Minimum Version | Verification Command |
|---|---|---|---|
| **Python** | `3.11.x` | `3.11.0` | `python3 --version` |
| **pip** | `24.x+` | `23.0` | `pip --version` |
| **Git** | `2.40+` | `2.28` | `git --version` |
| **Docker** | `24.0+` / `28.0+` | `20.10` | `docker --version` |
| **Docker Compose** | `v2.24+` | `v2.0` | `docker compose version` |
| **kubectl** *(Optional, for K8s)* | `1.30+` | `1.24` | `kubectl version --client` |
| **Helm** *(Optional, for K8s)* | `v3.14+` | `v3.8` | `helm version` |
| **Minikube** *(Optional, for local K8s)* | `v1.33+` | `v1.28` | `minikube version` |

> [!NOTE]
> All paths in this project are portable across macOS, Linux, and Windows using Python's `pathlib.Path` via `src/config.py`.

---

## 2. Quick Verification (Under 2 Minutes)

If you have Python 3.11 and Docker installed, run this rapid verification sequence:

```bash
# 1. Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 2. Install dependencies & HeartGuard package
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 3. Run all 29 automated tests (Data, Preprocessing, MLflow, Serving, Drift)
pytest -v

# 4. Verify Docker container build and live endpoints in an isolated sandbox
docker build -t heartguard-api:latest .
./scripts/test_docker.sh
```

---

## 3. Local Python Environment Setup

Clone the repository and enter the project root directory:

```bash
git clone https://github.com/bits-pilani-mlops/heart-disease-mlops.git
cd heart-disease-mlops   # Or your local directory
```

### Option A: Python Virtual Environment (`venv` - Recommended)

```bash
# 1. Create a clean virtual environment using Python 3.11
python3.11 -m venv .venv

# 2. Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate

# On Windows (cmd / PowerShell):
# .venv\Scripts\activate

# 3. Upgrade pip to latest version
pip install --upgrade pip setuptools wheel

# 4. Install all pinned dependencies
pip install -r requirements.txt

# 5. Install HeartGuard in editable mode (registers 'src' module cleanly)
pip install -e .

# 6. (Optional) Initialize local environment file
cp .env.example .env
```

### Option B: Conda Environment (`environment.yml`)

If you prefer Conda/Miniconda:

```bash
# 1. Create the environment from environment.yml
conda env create -f environment.yml

# 2. Activate the conda environment
conda activate mlops-heart-disease

# 3. Install HeartGuard package in editable mode
pip install -e .

# 4. (Optional) Initialize local environment file
cp .env.example .env
```

---

## 4. Data Acquisition, Cleaning & Baseline Profiling

The project includes automated scripts for ingestion, schema validation, and drift profiling. Execute these sequentially:

### Step 4.1: Download Raw Dataset
Downloads the official Cleveland Heart Disease dataset from the UCI Machine Learning Repository to `data/raw/heart.csv`:
```bash
python src/data/download_data.py
```
*Expected Output:* Confirms 303 rows downloaded and saved to `data/raw/heart.csv`.

### Step 4.2: Clean and Validate Data
Validates physiological schema ranges, formats target to binary (0 = absence, 1 = presence), and writes to `data/processed/heart_cleaned.csv`:
```bash
python src/data/clean_data.py
```
*Expected Output:*
- Total rows processed: 303
- Duplicate rows: 0
- Target distribution: 164 absence (54.1%), 139 disease (45.9%)
- Missing values retained for leak-free pipeline imputation: `ca` (4) and `thal` (2)

### Step 4.3: Generate Feature Distribution Baseline (For Drift Engine)
Calculates reference summary statistics (quantiles, bins, means, std) from the training data and saves to `models/baseline_distribution.json`:
```bash
python -c "from src.models.drift import generate_baseline_distribution; generate_baseline_distribution()"
```
*Expected Output:* `Baseline distribution profile written to models/baseline_distribution.json (13 features).`

---

## 5. Model Training & MLflow Experiment Tracking

The training pipeline uses Stratified 5-Fold Cross-Validation, optimizes hyperparameters via GridSearch, evaluates candidate algorithms (Logistic Regression, Random Forest, Gradient Boosting), logs all artifacts to MLflow, and serializes the winning model pipeline.

### Step 5.1: Execute Model Training Pipeline
```bash
export PYTHONPATH=.
python src/models/train.py
```

*What this does:*
1. Loads cleaned data and separates train (80%) and held-out test (20%) splits with stratification (`seed=42`).
2. Evaluates Logistic Regression, Random Forest, and Gradient Boosting under 5-Fold CV.
3. Automatically selects **L2-Regularized Logistic Regression** based on optimal ROC-AUC (**0.9632**) and high clinical recall (**89.29%** / only 3 false negatives).
4. Saves serialized Scikit-Learn pipeline to `models/model_pipeline.joblib` and `models/heartguard_pipeline.joblib`.
5. Logs parameters, metrics, confusion matrices, and ROC curves to the local MLflow backend (`mlruns/`).
6. Saves evaluation summary to `models/model_metrics.json` and `reports/model_comparison.json`.

### Step 5.2: Launch the MLflow Tracking UI
```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser to view:
- All experiment runs and fold comparisons
- Hyperparameters (`C`, `penalty`, `n_estimators`, `max_depth`)
- Metrics (`test_roc_auc`, `test_recall`, `test_accuracy`, `test_f1`, `cv_mean_roc_auc`)
- Logged artifacts (`confusion_matrix.png`, `roc_curve.png`, serialized MLflow model)

---

## 6. Automated Testing & Code Quality Assurance

The repository includes a comprehensive 29-test Pytest suite covering all modules with 100% pass rate.

### Step 6.1: Run All Tests
```bash
# Ensure your virtual environment is active (.venv)
pytest -v
```

*Test Coverage Breakdown (29/29 Passing):*
- **`tests/test_api.py`** (9 tests): Health probe, root metadata, valid prediction, boundary validation errors, batch predictions, drift status, drift evaluation, empty batch handling.
- **`tests/test_config.py`** (3 tests): Portable cross-platform paths, default settings, dynamic environment variable overrides.
- **`tests/test_data.py`** (4 tests): Schema validation, missing column errors, binary target encoding, zero-duplicate assertions.
- **`tests/test_drift.py`** (5 tests): Population Stability Index (PSI) calculation, Kolmogorov-Smirnov (KS) tests, baseline JSON structure, severe drift detection.
- **`tests/test_features.py`** (3 tests): `ColumnTransformer` pipeline, feature naming integrity, stratified train/test partitioning.
- **`tests/test_model.py`** (5 tests): Serialized artifact verification, single inference payload, missing-field handling, batch inference, performance metric computations.

### Step 6.2: Run Code Quality & Linting
```bash
ruff check .
```
*Expected Output:* `All checks passed!`

---

## 7. Local REST API Serving & Verification

The HeartGuard API is powered by FastAPI and Uvicorn, featuring sub-millisecond in-memory inference, strict Pydantic v2 physiological boundary validation, Prometheus metrics scraping, and a statistical drift evaluation engine.

### Starting the FastAPI Microservice
```bash
# Run locally with auto-reload enabled
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Server will start on **`http://localhost:8000`**.

### Interactive Web Interfaces
- **Interactive Web Diagnostic GUI:** [http://localhost:8000/app](http://localhost:8000/app)  
  *Allows clinicians or evaluators to test patient risk scoring interactively with preset presets.*
- **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc API Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Prometheus Metrics Scrape:** [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

### Testing Endpoints with cURL

#### 1. Liveness & Readiness Health Probe (`GET /health`)
```bash
curl -s http://localhost:8000/health | jq .
```
*Expected Response:*
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "timestamp": "2026-10-20T...",
  "uptime_seconds": 45.2
}
```

#### 2. Single Patient Prediction (`POST /predict`)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 67.0,
    "sex": 1.0,
    "cp": 4.0,
    "trestbps": 160.0,
    "chol": 286.0,
    "fbs": 0.0,
    "restecg": 2.0,
    "thalach": 108.0,
    "exang": 1.0,
    "oldpeak": 1.5,
    "slope": 2.0,
    "ca": 3.0,
    "thal": 3.0
  }' | jq .
```
*Expected Response:*
```json
{
  "prediction": 1,
  "label": "Heart Disease Risk Detected",
  "probability_disease": 0.9940,
  "confidence": 0.9940,
  "probabilities": {
    "absence": 0.0060,
    "presence": 0.9940
  },
  "latency_ms": 2.14,
  "model_version": "1.0.0"
}
```

#### 3. Batch Patient Prediction (`POST /predict/batch`)
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {
        "age": 45.0, "sex": 0.0, "cp": 2.0, "trestbps": 120.0, "chol": 190.0,
        "fbs": 0.0, "restecg": 0.0, "thalach": 172.0, "exang": 0.0, "oldpeak": 0.0,
        "slope": 1.0, "ca": 0.0, "thal": 3.0
      },
      {
        "age": 65.0, "sex": 1.0, "cp": 4.0, "trestbps": 150.0, "chol": 270.0,
        "fbs": 1.0, "restecg": 2.0, "thalach": 115.0, "exang": 1.0, "oldpeak": 2.2,
        "slope": 2.0, "ca": 2.0, "thal": 7.0
      }
    ]
  }' | jq .
```

#### 4. Covariate Drift Detection Evaluation (`POST /drift`)
```bash
curl -X POST http://localhost:8000/drift \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "age": 67.0, "sex": 1.0, "cp": 4.0, "trestbps": 160.0, "chol": 286.0,
        "fbs": 0.0, "restecg": 2.0, "thalach": 108.0, "exang": 1.0, "oldpeak": 1.5,
        "slope": 2.0, "ca": 3.0, "thal": 3.0
      }
    ]
  }' | jq .
```

#### 5. Scrape Prometheus Telemetry (`GET /metrics`)
```bash
curl -s http://localhost:8000/metrics | grep -E "heart_disease"
```

---

## 8. Docker Containerization Setup

The service is packaged using a multi-stage, security-hardened Dockerfile running as an unprivileged non-root user (`mlopsuser:mlopsgroup`).

### Building the Image
```bash
docker build -t heartguard-api:latest .
```

### Running the Container
```bash
docker run -d \
  --name heartguard-api \
  -p 8000:8000 \
  heartguard-api:latest
```

Verify the container is healthy:
```bash
# Check container status (look for '(healthy)')
docker ps --filter "name=heartguard-api"

# View container startup logs
docker logs --tail 20 heartguard-api

# Test endpoint inside container
curl -s http://localhost:8000/health
```

Stop and remove container:
```bash
docker stop heartguard-api && docker rm heartguard-api
```

### Automated Docker Verification Script
The repository provides an end-to-end verification script (`scripts/test_docker.sh`) that spins up the container, runs health, prediction, metrics, and drift checks, and cleanly tears it down:
```bash
chmod +x scripts/test_docker.sh
./scripts/test_docker.sh
```

---

## 9. Full-Stack Observability Stack (Docker Compose)

Launch the complete microservice alongside Prometheus and pre-configured Grafana:

```bash
docker compose up -d
```

### Stack Verification:
```bash
docker compose ps
```

| Service | Host Port | Internal Port | Description | Credentials |
|---|---|---|---|---|
| **API Server** | `8000` | `8000` | HeartGuard FastAPI Serving Engine | None |
| **Web GUI** | `8000/app` | `8000/app` | Diagnostic Clinical GUI | None |
| **Prometheus** | `9090` | `9090` | Time-Series Metrics Scraper | None |
| **Grafana** | `3001` | `3000` | Real-time Observability Dashboard | `admin` / `admin` |

- **Access Grafana Dashboard:** Navigate to **[http://localhost:3001](http://localhost:3001)**  
  *The custom dashboard `HeartGuard - Production Model Telemetry` is auto-provisioned, showing request rates, latency percentiles, error rates, and drift alerts.*

### Tear Down Stack:
```bash
docker compose down
```

---

## 10. Production Kubernetes & Helm Deployment

Deploy HeartGuard onto a local Kubernetes cluster (Minikube or Docker Desktop Kubernetes) or cloud provider (GKE, EKS, AKS).

### Kubernetes Raw Manifests (Minikube / Cluster)

#### Step 1: Start Minikube & Load Image
```bash
minikube start

# Load local Docker image into the Minikube registry
minikube image load heartguard-api:latest
```

#### Step 2: Apply Manifests
```bash
# 1. Create dedicated namespace
kubectl apply -f k8s/namespace.yaml

# 2. Deploy ConfigMap, Deployment, Service, Ingress, and HPA
kubectl apply -f k8s/
```

#### Step 3: Verify Deployment & Pod Status
```bash
# Check all resources in the mlops namespace
kubectl get all -n mlops

# Check Horizontal Pod Autoscaler (HPA)
kubectl get hpa -n mlops

# Check Ingress configuration
kubectl get ingress -n mlops
```

#### Step 4: Access Kubernetes Service Locally
```bash
# Forward port 8000 to the Kubernetes service
kubectl port-forward svc/heartguard-api-service 8000:80 -n mlops
```
Now access `http://localhost:8000/health` or `http://localhost:8000/app` in your browser.

---

### Helm Chart Deployment

HeartGuard includes a production Helm chart under `helm/heart-disease-api/` with parameterized values for replicas, autoscaling, ingress, and resources.

```bash
# 1. Install or upgrade release using Helm
helm install heartguard ./helm/heart-disease-api \
  --namespace mlops \
  --create-namespace

# 2. Inspect release status
helm status heartguard -n mlops

# 3. View deployed resources
kubectl get pods,svc,hpa -n mlops

# 4. Uninstall release when finished
helm uninstall heartguard -n mlops
```

---

## 11. Troubleshooting & FAQs

### Q1: `ModuleNotFoundError: No module named 'fastapi'` or `pytest` fails on collection
**Cause:** `pytest` is being executed by system Python instead of your virtual environment.  
**Fix:** Ensure your virtual environment is active:
```bash
source .venv/bin/activate
# Or call pytest directly from .venv:
.venv/bin/pytest -v
```

### Q2: Port 8000, 5000, 9090, or 3001 is already in use
**Cause:** Another service is bound to the target port.  
**Fix:** Identify and stop the occupying process, or pass an alternative port:
```bash
# For FastAPI:
uvicorn src.api.app:app --port 8080

# For MLflow:
mlflow ui --backend-store-uri file:./mlruns --port 5005

# For Docker:
docker run -p 8080:8000 heartguard-api:latest
```

### Q3: MLflow reports database connection or backend store warnings
**Cause:** Running `mlflow` from different directories with relative paths.  
**Fix:** Always run MLflow from the project root with the explicit backend flag:
```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

### Q4: Docker container fails with permission errors
**Cause:** The Dockerfile enforces a non-root user (`mlopsuser`, UID/GID `10001`).  
**Fix:** The Dockerfile automatically runs `chown -R mlopsuser:mlopsgroup /app`. Do not manually mount read-only local directories over `/app` without appropriate file permissions.

---

## 12. Assignment Deliverable Mapping

| Assignment Requirement | Relevant Scripts / Manifests | How to Verify |
|---|---|---|
| **1. Data Ingestion & EDA** | `src/data/download_data.py`, `src/data/clean_data.py`, `notebooks/01_heart_disease_eda.ipynb` | Run scripts; open EDA notebook or review `reports/figures/` |
| **2. Feature Pipeline & Model Tuning** | `src/features/build_features.py`, `src/models/train.py` | Run `python src/models/train.py` (CV ROC-AUC: **0.9632**) |
| **3. MLflow Experiment Tracking** | `src/models/train.py`, `mlruns/` | Run `mlflow ui --backend-store-uri file:./mlruns --port 5000` |
| **4. Packaging & Reproducibility** | `models/model_pipeline.joblib`, `pyproject.toml`, `requirements.txt`, `environment.yml` | `pip install -e .` and `pytest -v` |
| **5. CI/CD & Automated Testing** | `tests/` (29 tests), `.github/workflows/ci.yml`, `.github/workflows/train.yml` | `pytest -v` (100% passing) and `ruff check .` |
| **6. Docker Containerization** | `Dockerfile`, `scripts/test_docker.sh` | Run `./scripts/test_docker.sh` |
| **7. Production Deployment** | `k8s/` manifests, `helm/heart-disease-api/` | Apply manifests with `kubectl apply -f k8s/` or `helm install` |
| **8. Observability & Monitoring** | `monitoring/prometheus.yml`, `monitoring/grafana_dashboard.json`, `src/models/drift.py` | Run `docker compose up -d` and inspect `:3001` and `/drift` |
| **9. Documentation & Report** | `SETUP.md`, `README.md`, `reports/FINAL_REPORT.pdf`, `docs/implementation-audit.md` | Open `SETUP.md`, `README.md` or `reports/FINAL_REPORT.pdf` |
