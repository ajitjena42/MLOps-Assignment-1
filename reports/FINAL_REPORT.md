# HeartGuard — An Observable ML Pipeline
## End-to-End Enterprise MLOps Platform for Cardiovascular Risk Prediction

**Student ID:** 2025AE05839  
**Course:** AIMLCZG523 — Machine Learning Operations (MLOps)  
**Institution:** Birla Institute of Technology & Science (BITS), Pilani  
**Assignment:** Experimental Learning Assignment 01 (Total Marks: 50)  
**Dataset:** Heart Disease UCI (Cleveland Clinic Foundation Variant)  
**Date:** October 2026  
**Status:** Production Ready & Verified  

---

## Abstract

Cardiovascular diseases (CVDs) remain the leading contributor to global mortality, necessitating reliable, reproducible, and explainable computational risk stratification tools. This project delivers an enterprise-grade, end-to-end Machine Learning Operations (MLOps) system built on the clinical benchmark **Heart Disease UCI Cleveland dataset**. The architecture spans automated data ingestion with strict schema enforcement, exploratory data analysis across continuous and categorical biomarkers, leak-free feature preprocessing pipelines, dual candidate model optimization (Regularized Logistic Regression and Random Forest Ensembles), centralized experiment tracking via **MLflow**, automated unit/integration test suites (29 unit/integration tests, 100% pass rate), continuous integration and continuous deployment (**GitHub Actions**), isolated containerization (**Docker** with non-root security principles), container orchestration (**Kubernetes** manifests and production **Helm charts**), and telemetry instrumentation (**Prometheus** metrics and **Grafana** dashboards). Emphasizing clinical decision thresholds, the chosen production model achieves an outstanding **0.9632 ROC-AUC** and **89.29% Recall** on held-out test data, significantly curbing life-threatening false negatives while serving sub-6ms inferences.

---

## 1. Introduction & Clinical Problem Statement

Cardiovascular risk assessment requires balancing algorithmic discriminative capacity with operational safety. In diagnostic screening, **False Negatives (Type II errors)**—misclassifying an individual with coronary artery stenosis as disease-free—pose catastrophic clinical hazards, potentially delaying life-saving interventions such as statin therapy, coronary angioplasty, or bypass surgery. Conversely, **False Positives (Type I errors)** incur unnecessary anxiety and procedural healthcare expenses but do not immediately endanger patient survival.

The objective of this assignment is to operationalize an end-to-end MLOps pipeline that:
1. Translates raw multi-category angiographic records into a standardized, calibrated binary risk classification system.
2. Implements reproducible data cleaning, eliminating duplicates and managing missingness strictly within training folds.
3. Benchmarks multiple distinct model families (linear vs. non-linear ensemble methods) under Stratified 5-Fold Cross-Validation.
4. Governs experiment lifecycle, parameters, metrics, and models using MLflow.
5. Implements high-throughput, low-latency model serving via a containerized FastAPI application.
6. Deploys the service onto Kubernetes with Horizontal Pod Autoscaling (HPA) and ingress routing.
7. Continuously monitors system health, request throughput, latency distribution, and statistical class drift via Prometheus and Grafana.

---

## 2. Dataset Architecture, Attribution & Ethical Guidelines

### 2.1 Dataset Provenance & Attribution
The dataset utilized is the **Cleveland Clinic Foundation Heart Disease Dataset**, preserved and distributed by the University of California, Irvine (UCI) Machine Learning Repository:
- **Principal Investigators:**
  - Robert Detrano, M.D., Ph.D. (V.A. Medical Center, Long Beach and Cleveland Clinic Foundation)
  - Matthias Pfisterer, M.D. (University Hospital, Basel, Switzerland)
  - William Steinbrunn, M.D. (University Hospital, Zurich, Switzerland)
  - Andras Janosi, M.D. (Hungarian Institute of Cardiology, Budapest)
- **Donor:** David W. Aha (`aha@ics.uci.edu`)
- **Licensing:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Official Repository URL:** `https://archive.ics.uci.edu/dataset/45/heart+disease`

### 2.2 Feature Schema & Clinical Definitions
The dataset comprises 303 patient records across 14 standardized clinical attributes:

| Feature Name | Clinical Meaning | Data Type | Permissible Range / Values |
|---|---|---|---|
| `age` | Patient chronological age | Continuous | 29.0 – 77.0 years |
| `sex` | Biological sex | Binary | 1.0 = Male, 0.0 = Female |
| `cp` | Chest pain classification | Nominal | 1 = Typical Angina, 2 = Atypical Angina, 3 = Non-Anginal, 4 = Asymptomatic |
| `trestbps` | Resting systolic blood pressure on admission | Continuous | 94.0 – 200.0 mm Hg |
| `chol` | Serum cholesterol level | Continuous | 126.0 – 564.0 mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary | 1.0 = True, 0.0 = False |
| `restecg` | Resting electrocardiographic diagnosis | Nominal | 0 = Normal, 1 = ST-T Abnormality, 2 = Left Ventricular Hypertrophy |
| `thalach` | Maximum exercise heart rate achieved | Continuous | 71.0 – 202.0 bpm |
| `exang` | Exercise-induced angina | Binary | 1.0 = Yes, 0.0 = No |
| `oldpeak` | ST depression induced by exercise relative to rest | Continuous | 0.0 – 6.2 mm |
| `slope` | Slope of peak exercise ST segment | Ordinal | 1 = Upsloping, 2 = Flat, 3 = Downsloping |
| `ca` | Number of major coronary vessels colored by fluoroscopy | Ordinal | 0.0 – 3.0 vessels (4 missing records) |
| `thal` | Thalassemia scintigraphy defect status | Nominal | 3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect (2 missing records) |
| `target` (num) | Angiographic coronary artery diameter narrowing | Binary | 0 = Absence (<50% narrowing), 1 = Presence (>50% narrowing) |

### 2.3 Binary Target Formulation Justification
The original UCI recording scores coronary stenosis on an integer scale from 0 to 4:
- `0`: Angiographic diameter narrowing $<50\%$ (absence of hemodynamically significant stenosis).
- `1, 2, 3, 4`: Increasing stages of anatomical narrowing $>50\%$ (mild to multivessel critical disease).

Per standard clinical epidemiological consensus and assignment instructions, we map values $\{1, 2, 3, 4\} \to 1$ (Heart Disease Risk Detected) and $\{0\} \to 0$ (No Heart Disease Detected). This converts the dataset into a balanced binary prediction problem: **164 negative samples (54.1%)** and **139 positive samples (45.9%)**.

### 2.4 Ethical & Regulatory Statement
> **Clinical Disclaimer:** This machine learning software is implemented solely as an academic research demonstrator and educational project for the BITS Pilani MLOps course. It has not received FDA 510(k), CE mark, or Indian CDSCO regulatory clearance and must not be used as an autonomous medical diagnostic device or in place of professional medical evaluation.

---

## 3. Data Ingestion, Quality Validation & Cleaning

### 3.1 Automated Reproducible Ingestion
The module `src/data/download_data.py` automates raw retrieval from the canonical UCI Machine Learning Repository endpoint (`processed.cleveland.data`), backed by an automated fallback mirror. It validates line count, schema definitions, and column dimensions before committing to `data/raw/heart.csv`.

### 3.2 Data Quality & Missingness Protocol
- **Duplicate Records:** Duplicate analysis revealed **0 identical duplicate rows**, ensuring statistical independence across all patient vectors.
- **Missing Value Handling:** Missingness is restricted to 6 total values across two features: `ca` (4 missing values, 1.32%) and `thal` (2 missing values, 0.66%). 
- **Leakage Prevention Decision:** Rather than applying arbitrary global mean substitution to the entire dataset (which would induce test-set leakage), the missing values are retained as `NaN` in `data/processed/heart_cleaned.csv`. Imputation is deferred to the scikit-learn `Pipeline`, where imputers fit strictly on training folds during cross-validation.

---

## 4. Exploratory Data Analysis & Diagnostic Insights

Comprehensive exploratory analysis was executed through `notebooks/01_heart_disease_eda.ipynb`, producing 8 publication-grade visualization artifacts preserved in `reports/figures/`.

### 4.1 Target Class Distribution
The binary target formulation yields 164 absence cases (54.1%) and 139 disease cases (45.9%). This balance obviates the need for aggressive synthetic oversampling (such as SMOTE) while mitigating severe minority collapse.

### 4.2 Key Biomarker Correlations
Spearman rank correlation analysis revealed critical diagnostic associations:
1. **Chronotropic Incompetence (`thalach`):** Shows a strong negative correlation with disease presence ($\rho = -0.42$). Healthy subjects achieved an average maximum heart rate of 158.4 bpm under exercise treadmill testing, whereas cardiac patients plateaued at an average of 139.1 bpm due to impaired ventricular reserve.
2. **Exercise-Induced ST Depression (`oldpeak`):** Displays a strong positive correlation ($\rho = +0.42$). ST depression $>1.0$ mm serves as a classic electrocardiographic indicator of subendocardial myocardial ischemia.
3. **Fluoroscopy Vessel Obstruction (`ca`):** Patients exhibiting $\ge 1$ fluoroscopically opacified vessels exhibited disease prevalence in excess of $74\%$.
4. **The Angina Paradox (`cp`):** Counter-intuitively, chest pain category 4 (asymptomatic) demonstrated the highest prevalence of angiographic disease ($>72\%$), while category 1 (typical angina) demonstrated lower rates. In clinical practice, this reflects patient selection bias: asymptomatic patients are referred for invasive catheter angiography only when objective non-invasive testing (such as nuclear stress perfusion) reveals severe silent myocardial ischemia.

---

## 5. Feature Engineering & Preprocessing Pipeline

To eliminate data leakage, all preprocessing transformations are encapsulated within a unified scikit-learn `ColumnTransformer`:

```
Raw Patient Vector (13 Features)
       │
       ├──► Numerical Features (age, trestbps, chol, thalach, oldpeak)
       │         ├──► SimpleImputer(strategy='median')
       │         └──► StandardScaler()
       │
       └──► Categorical Features (sex, cp, fbs, restecg, exang, slope, ca, thal)
                 ├──► SimpleImputer(strategy='most_frequent')
                 └──► OneHotEncoder(handle_unknown='ignore', sparse_output=False)
```

1. **Numerical Stream:** Median imputation handles potential outlier skewness, followed by standard normalization ($\mu=0, \sigma=1$) to stabilize gradient descent and regularization penalties.
2. **Categorical Stream:** Mode imputation replaces missing categories, followed by one-hot encoding with `handle_unknown='ignore'` to guarantee resilience against novel categorical levels in production.
3. **Combined Dimensionality:** Transforms 13 raw clinical attributes into 27 encoded numerical features.

---

## 6. Model Development, Cross-Validation & Hyperparameter Tuning

### 6.1 Candidate Model Architectures
Two primary model architectures were developed and tuned, alongside a third gradient-boosted benchmark:
1. **L2-Regularized Logistic Regression:** Convex linear classifier providing well-calibrated posterior probabilities and direct log-odds coefficient interpretability.
2. **Random Forest Classifier:** Non-linear ensemble of decorrelated decision trees capable of capturing high-order feature interactions without parametric assumptions.
3. **Gradient Boosting Classifier:** Sequential boosting ensemble building successive trees along the negative gradient of cross-entropy loss.

### 6.2 Stratified Cross-Validation & Leak-Free Grid Search
- **Data Partitioning:** Stratified 80/20 train/test split (242 training instances, 61 test instances, random seed 42).
- **Cross-Validation:** 5-Fold Stratified K-Fold (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
- **Optimization Metric:** Models were tuned using `roc_auc` scoring to maximize discriminative separation across all decision thresholds.

### 6.3 Empirical Performance Results

| Model Architecture | 5-Fold CV ROC-AUC | Test Accuracy | Test Precision | Test Recall (Sensitivity) | Test F1-Score | Test ROC-AUC | Test False Negatives | Optimal Hyperparameters |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.9029** | **86.89%** | **83.33%** | **89.29%** | **0.8621** | **0.9632** | **3 / 28** | `C=5.0, penalty='l2', solver='lbfgs'` |
| **Random Forest** | **0.9027** | **88.52%** | **86.21%** | **89.29%** | **0.8772** | **0.9470** | **3 / 28** | `max_depth=5, min_leaf=2, min_split=5, n_est=100` |
| **Gradient Boosting** | **0.8847** | **86.89%** | **85.71%** | **85.71%** | **0.8571** | **0.9280** | **4 / 28** | `learning_rate=0.01, max_depth=3, n_est=100` |

### 6.4 Model Selection Trade-Off & Clinical Justification
Both Logistic Regression and Random Forest demonstrated identical recall on the held-out test cohort (89.29%, missing only 3 out of 28 cardiac patients). However, **Logistic Regression was selected as the primary production architecture** based on:
1. **Superior Discriminative Separation:** Achieved an outstanding test **ROC-AUC of 0.9632** (vs. 0.9470 for Random Forest), indicating superior separation across probability thresholds.
2. **Probability Calibration:** In healthcare screening, raw output probabilities must correspond to true clinical risk. Logistic Regression produces naturally calibrated sigmoidal probabilities without requiring Platt scaling.
3. **Inference Latency & Determinism:** Executes in under 1 millisecond per request, requiring minimal CPU footprint and zero memory overhead.
4. **Auditability & Explainability:** Coefficients directly translate to clinical odds ratios, satisfying healthcare algorithmic explainability criteria.

---

## 7. MLflow Experiment Tracking & Governance

Experiment tracking was integrated via **MLflow** (`src/models/train.py`), targeting the local repository store `file:./mlruns` under experiment `heart-disease-classification`:
- **Logged Parameters:** Estimator family, regularizers ($C$, learning rate, tree depth), split seeds, CV folds.
- **Logged Metrics:** Fold-wise ROC-AUC, test accuracy, precision, recall, F1, specificity, false negative count.
- **Logged Artifacts:** Confusion matrix plots, ROC comparison curves, feature importance charts, classification reports in JSON, and the fully fitted scikit-learn pipeline objects.
- **Reproducibility:** Run hashes record exact training conditions, enabling automated lineage auditing.

---

## 8. High-Performance Model Serving API (FastAPI)

The inference microservice was engineered in `src/api/app.py` using **FastAPI** and **Pydantic v2**:
- `GET /`: Returns service metadata, version, and route links.
- `GET /health`: Liveness and readiness probe reporting uptime and model loading status.
- `GET /metrics`: Standard Prometheus metrics endpoint.
- `POST /predict`: Single patient scoring with sub-millisecond feature transformation.
- `POST /predict/batch`: High-throughput array ingestion scoring up to 1,000 patient records in a single payload.
- **Validation:** Pydantic schemas enforce physiological boundaries (e.g., age $18-110$, blood pressure $70-260$, cholesterol $80-650$), rejecting corrupted requests with descriptive HTTP 422 errors.

---

## 9. Containerization, Kubernetes & Production Deployment

### 9.1 Multi-Stage Docker Packaging
The service is containerized via `Dockerfile` using `python:3.11-slim`:
- **Security Compliance:** Drops all root privileges, running under a dedicated non-root user `mlopsuser:mlopsgroup`.
- **Healthcheck:** Embedded `HEALTHCHECK` directive polls `curl -f http://localhost:8000/health`.
- **Image Optimization:** Pinned wheels and `.dockerignore` produce a compact, production-ready image.

### 9.2 Kubernetes Architecture
The application is specified for Kubernetes clusters (GKE, EKS, AKS, Minikube, Docker Desktop) in `k8s/`:
- **Deployment:** 3 replicas with `RollingUpdate` strategy (0 max unavailable).
- **Probes:** Configured Liveness, Readiness, and Startup probes.
- **Autoscaling (HPA):** Horizontal Pod Autoscaler dynamically scales from 2 to 8 pods when CPU exceeds 70% or memory exceeds 80%.
- **Networking:** Exposes a `LoadBalancer` service and an NGINX `Ingress` rule for host `heart-disease.bits-mlops.local`.

### 9.3 Production Helm Chart
The entire Kubernetes application is bundled into a configurable Helm chart under `helm/heart-disease-api/` with parameterized `values.yaml`, enabling single-command multi-environment deployments (`helm install heart-disease ./helm/heart-disease-api`).

---

## 10. Observability, Logging & Telemetry

1. **Structured Access Logging:** `src/api/middleware.py` generates structured JSON log entries containing timestamp, unique UUID `request_id`, client IP, HTTP method, route, status code, and latency in milliseconds.
2. **Statistical Data & Prediction Drift Engine (`/drift`):**
   - Implements baseline feature profiling computed from training data (`models/baseline_distribution.json`).
   - Evaluates two-sample Kolmogorov-Smirnov (KS) tests across continuous physiological biomarkers (`age`, `chol`, `trestbps`, `thalach`, `oldpeak`).
   - Calculates Population Stability Index (PSI) across categorical attributes (`sex`, `cp`, `restecg`, `exang`, `slope`, `ca`, `thal`).
   - Exposes `GET /drift` for baseline status and `POST /drift` for live batch drift evaluation.
3. **Prometheus Metrics Instrumentation:**
   - `heart_disease_api_requests_total`: Counter tracking throughput by method, endpoint, and status.
   - `heart_disease_predictions_total`: Counter tracking predictions categorized by diagnostic class.
   - `heart_disease_inference_latency_seconds`: Histogram measuring inference execution time.
   - `heart_disease_risk_probability`: Summary measuring statistical risk score distribution.
   - `heart_disease_model_loaded`: Gauge tracking in-memory model availability.
   - `heart_disease_drift_detected`: Gauge alerting when feature drift is statistically identified.
   - `heart_disease_drift_ratio`: Gauge reporting ratio of drifted features.
4. **Grafana Dashboard:** Configured in `monitoring/grafana_dashboard.json`, providing real-time telemetry panels for throughput, latency percentiles, error rates, and class distribution ratios.

---

## 11. CI/CD Automation & Quality Assurance (GitHub Actions)

A comprehensive GitHub Actions pipeline is codified in `.github/workflows/ci.yml`:
1. **Lint Stage:** Ruff static analysis enforcing strict code quality and formatting.
2. **Test Stage:** Executes all 29 Pytest unit and integration tests across data cleaning, transformers, model inference, drift detection, configuration, and FastAPI routes (100% pass rate).
3. **Train & Log Stage:** Runs data download, cleaning, model training, MLflow tracking, and archives pipeline artifacts.
4. **Container Build Stage:** Builds the Docker image, starts the container in background, executes live curl health checks, and verifies zero regression.

---

## 12. Conclusion & Operational Summary

This project establishes a rigorous, production-grade MLOps platform demonstrating every phase of the modern machine learning lifecycle. By combining leak-free scikit-learn preprocessing, MLflow tracking, automated testing, non-root Docker security, Kubernetes auto-scaling, and Prometheus telemetry, the system achieves an optimal balance between academic rigor and enterprise production readiness.

---

## Deliverables & Artifact Index

| Deliverable Item | Location in Repository | Verification Proof |
|---|---|---|
| Cleaned Dataset & Ingestion Script | `data/processed/heart_cleaned.csv`, `src/data/download_data.py` | 303 rows, verified schema |
| Jupyter EDA Notebook & Figures | `notebooks/01_heart_disease_eda.ipynb`, `reports/figures/` | 8 figures generated at 300 DPI |
| Training Code & Model Pipeline | `src/models/train.py`, `models/model_pipeline.joblib` | Saved pipeline, AUC = 0.9632 |
| MLflow Tracking Repository | `mlruns/`, `reports/model_comparison.json` | 3 candidate model runs logged |
| Test Suite (29 Unit & Integration Tests) | `tests/test_data.py`, `tests/test_features.py`, `tests/test_model.py`, `tests/test_api.py`, `tests/test_drift.py`, `tests/test_config.py` | 29 passed in 2.88s (100% pass rate) |
| CI/CD Pipeline Workflows | `.github/workflows/ci.yml`, `.github/workflows/train.yml` | Quality gates & automated testing |
| FastAPI Serving & Drift API | `src/api/app.py`, `src/api/schemas.py`, `src/models/drift.py` | Live /predict, /health, /metrics, /drift |
| Dockerfile & Compose Stack | `Dockerfile`, `docker-compose.yml`, `scripts/test_docker.sh` | Non-root security, live container run |
| Kubernetes Manifests & Helm Chart | `k8s/`, `helm/heart-disease-api/` | Deployment, Service, HPA, Ingress |
| Observability Stack | `monitoring/prometheus.yml`, `monitoring/grafana_dashboard.json` | Active metrics scraping & dashboards |
| Final Academic Report (MD & PDF) | `reports/FINAL_REPORT.md`, `reports/FINAL_REPORT.pdf` | Comprehensive 10-page document |
| Visual Diagrams & Screenshots | `reports/figures/architecture_diagram.png`, `reports/screenshots/` | 7 high-res PNG artifacts |
