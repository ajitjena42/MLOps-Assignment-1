# HeartGuard Implementation Audit & Architecture Assessment

**Project:** HeartGuard — An Observable ML Pipeline  
**Student ID:** 2025AE05839  
**Course:** AIMLCZG523 — Machine Learning Operations (MLOps)  
**Institution:** Birla Institute of Technology & Science (BITS), Pilani  
**Date:** October 2026  
**Audit Status:** Completed  

---

## 1. Executive Summary & Audit Context

This audit evaluates the baseline implementation of an end-to-end Machine Learning Operations (MLOps) project for cardiovascular disease risk prediction using the canonical Cleveland Clinic Foundation dataset from the UCI Machine Learning Repository. 

The baseline codebase possesses a structured foundation, including scikit-learn preprocessing pipelines, MLflow experiment logging, a FastAPI model serving layer, Docker packaging, Kubernetes manifests, a Helm chart, and Prometheus/Grafana monitoring configurations. However, a rigorous code, configuration, and runtime audit revealed multiple critical architectural discrepancies, broken dependencies, unhandled runtime failures, security/privacy leaks, and gaps between what was documented versus what was actually implemented.

This document details the findings from the audit and establishes an engineering roadmap for high-impact, academically defensible improvements.

---

## 2. Directory Tree & Component Inventory

```
.
├── .dockerignore
├── .env.example                               # Environment variable reference template
├── .github/
│   └── workflows/
│       ├── ci.yml                             # GitHub Actions CI workflow (lint, test, smoke test)
│       └── train.yml                          # Retraining and model quality gate workflow
├── .gitignore
├── Dockerfile                                 # Multi-stage / slim Python 3.11 container manifest
├── README.md                                  # Repository documentation
├── data/
│   ├── raw/heart.csv                          # Ingested raw UCI Cleveland dataset (303 rows)
│   └── processed/heart_cleaned.csv            # Cleaned, schema-validated dataset with binary target
├── docker-compose.yml                         # Full-stack composition (API + Prometheus + Grafana)
├── environment.yml                            # Conda environment definition
├── helm/
│   └── heart-disease-api/                     # Helm chart (Deployment, Service, Ingress, HPA, ConfigMap)
├── k8s/                                       # Raw Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   └── hpa.yaml
├── models/
│   ├── model_pipeline.joblib                  # Serialized Scikit-Learn production pipeline
│   ├── model_pipeline.pkl                     # Standard pickle serialization
│   └── mlflow_model/                          # MLflow pyfunc model bundle with MLmodel & conda env
├── monitoring/
│   ├── prometheus.yml                         # Scrape targets and interval settings
│   ├── grafana_dashboard.json                 # Dashboard definition with 6 visual panels
│   └── provisioning/                          # Auto-provisioning for datasources and dashboards
├── notebooks/
│   ├── 01_heart_disease_eda.ipynb             # Exploratory Data Analysis notebook
│   └── heart_disease_master.ipynb             # Master pipeline notebook
├── reports/
│   ├── FINAL_REPORT.md                        # Academic final report markdown
│   ├── FINAL_REPORT.pdf                       # Compiled 10-page publication-style report
│   ├── figures/                               # EDA charts, ROC curves, confusion matrices
│   ├── screenshots/                           # Verification evidence screenshots
│   ├── model_comparison.json                  # Cross-validation and test set evaluation metrics
│   └── Heart_Disease_API.postman_collection.json # API testing collection
├── requirements.txt                           # Pinned Python package dependencies
├── scripts/
│   ├── compile_report_pdf.py                  # ReportLab PDF compilation script
│   ├── generate_architecture_and_screenshots.py # Visual asset generator
│   ├── generate_pipeline_video.py             # Demonstration video generator
│   ├── build_and_run_eda_notebook.py          # Notebook execution utility
│   └── test_docker.sh                         # Container automated verification script
├── src/
│   ├── data/
│   │   ├── download_data.py                   # Automated data ingestion with failover mirror
│   │   └── clean_data.py                      # Schema validation, missingness handling, binarization
│   ├── features/
│   │   └── build_features.py                  # ColumnTransformer feature pipeline (imputation, scaling, OHE)
│   ├── models/
│   │   ├── train.py                           # 5-fold CV hyperparameter search and MLflow tracking
│   │   ├── evaluate.py                        # Metric calculations and diagnostic plotting
│   │   └── predict.py                         # Standalone inference engine
│   └── api/
│       ├── app.py                             # FastAPI serving application with Prometheus instrumentation
│       ├── schemas.py                         # Pydantic v2 physiological boundary validation models
│       ├── middleware.py                      # Request correlation ID and structured JSON access logger
│       └── templates/index.html               # Interactive browser diagnostic GUI
└── tests/
    ├── conftest.py
    ├── test_data.py                           # Schema and cleaning unit tests
    ├── test_features.py                       # Preprocessor and stratification unit tests
    ├── test_model.py                          # Pipeline prediction and calibration unit tests
    └── test_api.py                            # FastAPI endpoint integration tests
```

---

## 3. End-to-End MLOps Pipeline Flow

The intended end-to-end operational pipeline progresses through the following sequential stages:

```
[UCI Repository] 
       │
       ▼
1. Data Ingestion (src/data/download_data.py)
   - Fetches 303 patient records from UCI repository with fallback mirror.
       │
       ▼
2. Data Cleaning & Validation (src/data/clean_data.py)
   - Enforces 14-column clinical schema; validates physiological bounds.
   - Converts multi-class angiographic score (0-4) into binary target: 0 (absence) vs 1 (presence).
   - Preserves missing entries ('ca', 'thal') as NaN to prevent cross-validation data leakage.
       │
       ▼
3. Feature Engineering & Column Transformation (src/features/build_features.py)
   - Implements ColumnTransformer: Median imputation + StandardScaler for 5 continuous biomarkers;
     Mode imputation + OneHotEncoder(handle_unknown='ignore') for 8 categorical biomarkers.
       │
       ▼
4. Model Training & Governance (src/models/train.py)
   - Stratified 5-Fold Cross-Validation on 80% train split (242 instances), held-out test split (61 instances).
   - Tunes Logistic Regression (L2), Random Forest, and Gradient Boosting.
   - Logs hyperparameter spaces, CV scores, test metrics, confusion matrices, and ROC curves to MLflow.
   - Selection criterion: Weighted clinical score prioritizing Recall (60%) and ROC-AUC (40%). Winner: Logistic Regression (AUC: 0.9632, Recall: 89.29%).
       │
       ▼
5. Model Serialization (models/)
   - Exports unified pipeline (preprocessor + classifier) to joblib, pickle, and MLflow Model format.
       │
       ▼
6. Inference Serving (src/api/app.py)
   - Exposes REST endpoints (`/predict`, `/predict/batch`, `/health`, `/metrics`, `/app`).
   - Pydantic v2 boundary validation checks inputs.
   - Prometheus metrics observe request latency, prediction classes, and risk probabilities.
       │
       ▼
7. Containerization & Orchestration (Dockerfile, docker-compose.yml, k8s/, helm/)
   - Python 3.11 slim image with non-root security user (`mlopsuser`).
   - Kubernetes 3-replica rolling deployment with liveness/readiness probes, HPA, and Service/Ingress.
   - Prometheus scrapes `/metrics`; Grafana visualizes throughput, latency, and diagnosis distributions.
```

---

## 4. Implementation Status: Implemented vs. Documented vs. Gaps

| Capability / Feature | Status | Evidence / Analysis |
|---|---|---|
| **Data Ingestion** | Implemented | `src/data/download_data.py` cleanly downloads and verifies raw records with fallback. |
| **Data Validation & Cleaning** | Implemented | `src/data/clean_data.py` enforces schema, range checks, and binary binarization. |
| **Zero-Leakage Transformation** | Implemented | `src/features/build_features.py` builds `ColumnTransformer` fitted strictly on train folds. |
| **Model Comparison & MLflow** | Implemented | `src/models/train.py` evaluates 3 models with Stratified 5-Fold CV and logs runs to MLflow. |
| **API Serving (`/predict`, `/predict/batch`)** | Implemented | FastAPI endpoints serve predictions with Pydantic validation. |
| **Health Probe (`/health`)** | Partially Implemented | Exists, but redundant and fragile: re-loads model artifact on *every* health check call. |
| **Drift Monitoring (`/drift`)** | **Documented Only / Missing** | Documented in `FINAL_REPORT.md` and referenced in test cache (`test_drift_endpoint`), but **not implemented** in `app.py`. No statistical drift calculation engine exists. |
| **Centralized Configuration** | **Fragile / Missing** | Settings and paths are hardcoded across multiple files. `.env.example` exists, but application modules do not read environment variables systematically. |
| **Model Deserialization Compatibility** | **Broken in Baseline** | The downloaded `model_pipeline.joblib` was pickled with scikit-learn 1.5.2, throwing `AttributeError: '_RemainderColsList'` under modern scikit-learn runtime (1.6+). |
| **Error Handling & Privacy** | **Fragile** | API error handlers leak raw exception details (`f"Inference execution failure: {str(e)}"`), exposing internal stack details and file paths. |
| **Docker Containerization** | Implemented | Valid Dockerfile using `python:3.11-slim`, non-root user `mlopsuser`, and health check. |
| **Kubernetes & Helm Deployment** | Implemented | Manifests and Helm templates are well-structured; local Minikube cluster is active and operational. |
| **Observability (Prometheus/Grafana)** | Implemented | Metrics match Grafana panels; live Grafana and Prometheus containers are currently running on Docker. |

---

## 5. Current Git State & Revision Tracking

- **Git Status:** Git initialized on branch `main`.
- **Baseline Commit Created:** Root commit `f85328d` committed with message `"Baseline: original downloaded assignment project"` and tagged as `original-download`.
- **Customization Branch:** Created and checked out isolated branch `feature/heartguard-customization` for all modifications.
- **Original Source Integrity:** Preserved in pristine state under git tag `original-download`.

---

## 6. Detailed Technical Audit Findings

### Finding 1: Broken Model Serialization & Scikit-Learn Version Mismatch
- **Issue:** Running `pytest` immediately after download resulted in 6 failed tests due to:
  ```
  AttributeError: Can't get attribute '_RemainderColsList' on <module 'sklearn.compose._column_transformer' ...>
  ```
- **Root Cause:** The downloaded binary artifact `models/model_pipeline.joblib` was generated using scikit-learn 1.5.2. In newer scikit-learn versions, the internal private class `_RemainderColsList` was refactored.
- **Resolution:** Re-executing `src/models/train.py` trained all models deterministically (seed `42`), generated fresh MLflow runs, and produced compatible `model_pipeline.joblib`, `model_pipeline.pkl`, and `models/mlflow_model/` artifacts. All 18 tests passed immediately.
- **Action Required:** Ensure training scripts can be executed seamlessly and that model loading includes robust fallback and validation.

### Finding 2: Hardcoded Absolute Machine Paths Across Scripts
- **Issue:** Three critical automation scripts contained hardcoded absolute filesystem paths:
  - `scripts/compile_report_pdf.py`: hardcoded absolute machine path
  - `scripts/generate_pipeline_video.py`: hardcoded absolute machine path
  - `scripts/build_and_run_eda_notebook.py`: hardcoded absolute machine path
- **Impact:** Any attempt to execute these scripts on a different machine fails immediately with `FileNotFoundError` or writes outside the project directory.
- **Action Required:** Refactor all workspace path resolutions to use dynamic relative resolution: `Path(__file__).resolve().parents[...]`.

### Finding 3: Missing Statistical Drift Detection Endpoint (`/drift`)
- **Issue:** Both the assignment guidance and the report documentation highlight real-time concept and statistical feature drift monitoring. Furthermore, test node ID caches showed a previous `test_drift_endpoint`. However, `src/api/app.py` has no `/drift` endpoint.
- **Action Required:** Build a dedicated, statistically sound drift detection module and endpoint (`/drift`). Calculate a baseline distribution from training data (means, variances, quantiles for numerical features; frequencies for categorical features), implement statistical tests (two-sample Kolmogorov-Smirnov test for continuous features; population stability / total variation distance for categorical features), and expose drift status via REST and Prometheus metrics.

### Finding 4: Inefficient Model Reloading on Health Probes & Deprecated Startup Event
- **Issue:** In `src/api/app.py`, the `/health` endpoint executes `pipeline = load_model(MODEL_PATH)` on every single invocation. When Kubernetes liveness/readiness probes run every 5-10 seconds, this introduces unnecessary CPU cycles and filesystem status checks. Additionally, `app.on_event("startup")` is deprecated in FastAPI.
- **Action Required:** Modernize FastAPI lifecycle to use the `lifespan` context manager. Load the pipeline once during application startup, attach the loaded model and metadata to `app.state`, and have `/health` and `/predict` retrieve the instance directly from application state.

### Finding 5: Lack of Centralized Configuration & Inconsistent Artifact Names
- **Issue:** Environment variables defined in `.env.example` (`MODEL_PATH=models/heartguard_pipeline.joblib`) differ from the hardcoded path used in code (`models/model_pipeline.joblib`). GitHub Actions workflows in `.github/workflows/ci.yml` and `train.yml` point to non-existent script paths like `scripts/train.py` and `scripts/download_data.py`.
- **Action Required:** Create a centralized configuration module (`src/config.py`) using Pydantic Settings / `os.getenv` with sensible defaults. Synchronize artifact paths across `.env.example`, code, Docker, Kubernetes, and CI workflows.

### Finding 6: Sensitive Error Exposure in API Responses
- **Issue:** In `src/api/app.py`, exceptions in `/predict` and `/predict/batch` return `detail=f"Inference execution failure: {str(e)}"`. If an unhandled error occurs (e.g., path resolution, unpickling, schema mismatch), internal file paths and stack information are leaked to clients.
- **Action Required:** Sanitize error responses with a standardized generic error message, while logging full stack traces internally to secure server logs with correlation IDs.

### Finding 7: Privacy & Academic Submission Metadata Cleanup
- **Issue:** Multiple artifacts contain personal identifying information or outdated placeholder metadata:
  - Notebooks contain output cell text referencing historical machine paths.
  - `helm/heart-disease-api/Chart.yaml` contained generic placeholder student email.
  - `README.md` and `reports/FINAL_REPORT.md` require project name **HeartGuard — An Observable ML Pipeline** and student ID `2025AE05839`.
- **Action Required:** Systematically sanitize personal names and machine paths from notebooks, charts, and documentation, ensuring consistent academic attribution.

---

## 7. Proposed Meaningful Improvements

Based on our findings, we propose 5 meaningful, technically justified improvements that elevate the engineering rigor without altering the core scope:

1. **Centralized, Portable Configuration Management (`src/config.py`)**:
   - Centralize all environment variables, model paths, server settings, and monitoring thresholds into a typed configuration layer.
   - Synchronize configuration across `.env.example`, Docker, Kubernetes, and CI/CD pipelines.

2. **Statistical Data & Prediction Drift Detection Engine (`/drift`)**:
   - Implement baseline feature profiling computed from training data (`models/baseline_distribution.json`).
   - Implement Kolmogorov-Smirnov (KS) tests for continuous biomarkers (`age`, `chol`, `trestbps`, `thalach`, `oldpeak`) and Population Stability Index (PSI) / Total Variation distance for categorical biomarkers.
   - Expose `/drift` endpoint with feature-level p-values, drift flags, and Prometheus drift metrics.

3. **Modernized FastAPI Lifespan & High-Efficiency State Management**:
   - Replace deprecated `@app.on_event` with async `lifespan` handler.
   - Store loaded model pipeline, metadata, and baseline distributions on `app.state`.
   - Prevent redundant filesystem loading on `/health` probes.

4. **Production Hardened Error Handling & Secure API Contracts**:
   - Sanitize all API exceptions to prevent internal path and stack leakage.
   - Expand Pydantic validation schemas to return informative, safe clinical constraint feedback.

5. **Expanded Automated Test Suite & Code Quality Gates**:
   - Add comprehensive tests for `/drift`, configuration overrides, invalid inputs, edge cases, and model loading failures.
   - Format and lint codebase to pass Ruff checks cleanly.

---

## 8. Verification Strategy & Acceptance Criteria

- **Unit & Integration Testing:** All existing and new tests must pass with 100% pass rate (`pytest -v`).
- **Linter Compliance:** Zero syntax or import errors under `ruff check`.
- **Model Training & MLflow:** Training runs end-to-end; artifacts verified under `models/` and `mlruns/`.
- **Docker Containerization:** Build image `heartguard-api:latest`; verify container starts, passes `/health`, and serves `/predict` and `/drift`.
- **Kubernetes Deployment:** Load image into local Minikube cluster; verify deployment, service, probes, and HPA in namespace `mlops`.
- **Monitoring Integration:** Verify Prometheus metrics endpoint `/metrics` exposes request, latency, risk probability, and drift metrics.
- **Reporting:** Recompile `reports/FINAL_REPORT.pdf` with updated metadata, student ID `2025AE05839`, and project title.
