"""Model training, hyperparameter tuning, evaluation, and MLflow experiment tracking.

Trains Logistic Regression, Random Forest, and Gradient Boosting pipelines using
Stratified 5-Fold cross-validation and evaluates on a held-out test set.
Logs all parameters, metrics, plots, and fitted models to MLflow.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MPL_DIR = PROJECT_ROOT / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from typing import Any

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.features.build_features import (
    get_feature_names,
    get_preprocessor,
    prepare_train_test_data,
)
from src.models.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_model")

RANDOM_STATE = 42
CV_SPLITS = 5
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_cleaned.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLFLOW_TRACKING_URI = f"file:{PROJECT_ROOT / 'mlruns'}"
EXPERIMENT_NAME = "heart-disease-classification"


def get_candidate_models() -> dict[str, tuple[Any, dict[str, Any]]]:
    """Define candidate estimators and their hyperparameter search spaces."""
    return {
        "LogisticRegression": (
            LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
            {
                "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["lbfgs", "liblinear"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [50, 100, 150],
                "classifier__max_depth": [3, 5, 8, None],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [50, 100],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__max_depth": [2, 3, 4],
            },
        ),
    }


def ensure_portable_mlruns(project_root: Path) -> None:
    """Ensure all MLflow experiment metadata files use portable paths rather than hardcoded machine paths."""
    mlruns_dir = project_root / "mlruns"
    if not mlruns_dir.exists():
        return
    import yaml

    for meta_path in mlruns_dir.rglob("meta.yaml"):
        try:
            content = meta_path.read_text(encoding="utf-8")
            if "/Users/" in content or "file:///" in content:
                data = yaml.safe_load(content)
                changed = False
                for key in ["artifact_location", "artifact_uri"]:
                    if key in data and isinstance(data[key], str) and "mlruns" in data[key]:
                        idx = data[key].find("mlruns")
                        data[key] = data[key][idx:]
                        changed = True
                if changed:
                    meta_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not sanitize {meta_path}: {e}")

    for mlmodel_path in mlruns_dir.rglob("MLmodel"):
        try:
            content = mlmodel_path.read_text(encoding="utf-8")
            if "/Users/" in content or "file:///" in content:
                data = yaml.safe_load(content)
                if (
                    "artifact_path" in data
                    and isinstance(data["artifact_path"], str)
                    and "mlruns" in data["artifact_path"]
                ):
                    idx = data["artifact_path"].find("mlruns")
                    data["artifact_path"] = data["artifact_path"][idx:]
                    mlmodel_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not sanitize {mlmodel_path}: {e}")


def train_and_track(source_name: str = "notebooks/heart_disease_master.ipynb"):
    """Execute complete training workflow with MLflow tracking and artifact generation."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset missing at {DATA_PATH}. Run src/data/clean_data.py first.")

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} rows from {DATA_PATH}")

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = prepare_train_test_data(
        df, target_col="target", test_size=0.2, random_state=RANDOM_STATE
    )
    logger.info(f"Train split: {X_train.shape[0]} samples; Test split: {X_test.shape[0]} samples")
    logger.info(f"Train label balance: {y_train.value_counts().to_dict()}")
    logger.info(f"Test label balance: {y_test.value_counts().to_dict()}")

    # Setup MLflow
    ensure_portable_mlruns(PROJECT_ROOT)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}; Experiment: {EXPERIMENT_NAME}")

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    candidate_models = get_candidate_models()

    best_overall_score = -1.0
    best_overall_pipeline = None
    best_overall_name = None
    models_comparison = {}
    roc_plot_data = {}
    trained_pipelines = {}

    for name, (estimator, param_grid) in candidate_models.items():
        logger.info(f"\n================ Training Candidate: {name} ================")
        run_tags = {
            "mlflow.source.name": source_name,
            "mlflow.source.type": "NOTEBOOK" if source_name.endswith(".ipynb") else "LOCAL",
            "notebook": "heart_disease_master.ipynb",
        }
        with mlflow.start_run(run_name=name, tags=run_tags) as run:
            # Construct unified pipeline
            pipeline = Pipeline(steps=[("preprocessor", get_preprocessor()), ("classifier", estimator)])

            # Hyperparameter Tuning using Stratified Cross-Validation
            # Scoring on 'roc_auc' balances sensitivity and specificity across thresholds
            grid_search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
                return_train_score=True,
            )

            grid_search.fit(X_train, y_train)
            best_pipe = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_roc_auc = float(grid_search.best_score_)
            logger.info(f"{name} - Best Cross-Validation ROC-AUC: {cv_roc_auc:.4f}")
            logger.info(f"{name} - Optimal Hyperparameters: {best_params}")

            # Evaluate on held-out test set
            y_pred = best_pipe.predict(X_test)
            y_proba = best_pipe.predict_proba(X_test)[:, 1]

            test_metrics = compute_metrics(y_test.values, y_pred, y_proba)
            logger.info(
                f"{name} - Held-Out Test Results: Accuracy={test_metrics['accuracy']:.4f}, "
                f"Recall={test_metrics['recall']:.4f}, Precision={test_metrics['precision']:.4f}, "
                f"F1={test_metrics['f1']:.4f}, ROC-AUC={test_metrics['roc_auc']:.4f}, "
                f"False Negatives={test_metrics['false_negatives']}"
            )

            # Store pipeline and ROC curve data
            trained_pipelines[name] = best_pipe
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_plot_data[name] = (fpr, tpr, test_metrics["roc_auc"])

            # Diagnostic plots
            cm_path = FIGURES_DIR / f"confusion_matrix_{name.lower()}.png"
            plot_confusion_matrix(y_test.values, y_pred, name, cm_path)

            # Feature importance / Coefficients
            fitted_preprocessor = best_pipe.named_steps["preprocessor"]
            feature_names = get_feature_names(fitted_preprocessor)
            clf = best_pipe.named_steps["classifier"]
            fi_path = FIGURES_DIR / f"feature_importance_{name.lower()}.png"
            if hasattr(clf, "feature_importances_"):
                plot_feature_importance(feature_names, clf.feature_importances_, name, fi_path, "Gini Importance")
            elif hasattr(clf, "coef_"):
                plot_feature_importance(feature_names, clf.coef_[0], name, fi_path, "Log-Odds Coefficients")

            # Classification report
            cls_report = classification_report(y_test, y_pred, output_dict=True)

            # MLflow Logging
            # 1. Parameters
            mlflow.log_params(best_params)
            mlflow.log_params(
                {
                    "model_type": name,
                    "cv_folds": CV_SPLITS,
                    "random_state": RANDOM_STATE,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "scoring_metric": "roc_auc",
                }
            )

            # 2. Metrics
            mlflow.log_metric("cv_best_roc_auc", cv_roc_auc)
            for metric_k, metric_v in test_metrics.items():
                if metric_v is not None and isinstance(metric_v, (int, float)):
                    mlflow.log_metric(f"test_{metric_k}", metric_v)

            # 3. Artifacts
            mlflow.log_artifact(str(cm_path))
            if fi_path.exists():
                mlflow.log_artifact(str(fi_path))

            report_file = REPORTS_DIR / f"classification_report_{name.lower()}.json"
            with open(report_file, "w") as f:
                json.dump(cls_report, f, indent=2)
            mlflow.log_artifact(str(report_file))

            # 4. Log Model with example signature
            input_example = X_test.iloc[:2]
            mlflow.sklearn.log_model(
                sk_model=best_pipe,
                artifact_path="pipeline",
                input_example=input_example,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

            # Record model performance summary formatted for reporting
            models_comparison[name] = {
                "5-Fold CV ROC-AUC": round(cv_roc_auc, 4),
                "Test Accuracy": f"{test_metrics['accuracy'] * 100:.2f}%",
                "Test Precision": f"{test_metrics['precision'] * 100:.2f}%",
                "Test Recall (Sensitivity)": f"{test_metrics['recall'] * 100:.2f}%",
                "Test F1-Score": round(test_metrics["f1"], 4),
                "Test ROC-AUC": round(test_metrics["roc_auc"], 4),
                "False Negatives": f"{test_metrics['false_negatives']} / 28",
                "Best Parameters": str(best_params),
                "mlflow_run_id": run.info.run_id,
            }

            # Decision criteria: Recall is weighted heavily in clinical diagnosis to minimize False Negatives
            # We select model with top combination of Recall and ROC-AUC
            selection_score = (test_metrics["recall"] * 0.6) + (test_metrics["roc_auc"] * 0.4)
            if selection_score > best_overall_score:
                best_overall_score = selection_score
                best_overall_pipeline = best_pipe
                best_overall_name = name

    # Combined ROC Plot across all candidate models
    combined_roc_path = FIGURES_DIR / "roc_curve.png"
    plot_roc_curves(roc_plot_data, combined_roc_path)
    logger.info(f"Saved combined ROC curve to {combined_roc_path}")

    # Best Model Confusion Matrix copy to standard path
    best_cm_source = FIGURES_DIR / f"confusion_matrix_{best_overall_name.lower()}.png"
    standard_cm_dest = FIGURES_DIR / "confusion_matrix.png"
    if best_cm_source.exists():
        import shutil

        shutil.copy(best_cm_source, standard_cm_dest)

    # Save final chosen model pipeline across reusable formats (Joblib, Pickle, MLflow Package)
    # 1. Joblib Format (primary and HeartGuard alias)
    saved_model_path = MODELS_DIR / "model_pipeline.joblib"
    joblib.dump(best_overall_pipeline, saved_model_path)
    joblib.dump(best_overall_pipeline, MODELS_DIR / "heartguard_pipeline.joblib")

    # 2. Standard Pickle Format
    import pickle

    saved_pickle_path = MODELS_DIR / "model_pipeline.pkl"
    with open(saved_pickle_path, "wb") as f:
        pickle.dump(best_overall_pipeline, f)
    with open(MODELS_DIR / "heartguard_pipeline.pkl", "wb") as f:
        pickle.dump(best_overall_pipeline, f)

    # 3. Standalone MLflow Model Package (with schema signature & Conda/Pip requirements)
    from mlflow.models.signature import infer_signature

    mlflow_model_dir = MODELS_DIR / "mlflow_model"
    if mlflow_model_dir.exists():
        import shutil

        shutil.rmtree(mlflow_model_dir)
    input_sample = X_test.iloc[:2]
    signature = infer_signature(input_sample, best_overall_pipeline.predict(input_sample))
    mlflow.sklearn.save_model(
        sk_model=best_overall_pipeline,
        path=str(mlflow_model_dir),
        signature=signature,
        input_example=input_sample,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )

    # 4. Save Model Metrics for Automated CI/CD Quality Gates
    winning_metrics = {
        "model_name": best_overall_name,
        "accuracy": round(float(models_comparison[best_overall_name]["Test Accuracy"].replace("%", "")) / 100.0, 4),
        "precision": round(float(models_comparison[best_overall_name]["Test Precision"].replace("%", "")) / 100.0, 4),
        "recall": round(
            float(models_comparison[best_overall_name]["Test Recall (Sensitivity)"].replace("%", "")) / 100.0, 4
        ),
        "f1": models_comparison[best_overall_name]["Test F1-Score"],
        "roc_auc": models_comparison[best_overall_name]["Test ROC-AUC"],
        "cv_best_roc_auc": models_comparison[best_overall_name]["5-Fold CV ROC-AUC"],
    }
    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(winning_metrics, f, indent=2)

    # 5. Generate and persist baseline distribution profile for drift detection
    from src.models.drift import generate_baseline_distribution

    generate_baseline_distribution(DATA_PATH, MODELS_DIR / "baseline_distribution.json")

    logger.info("\n=======================================================")
    logger.info(f"Winning Architecture: {best_overall_name}")
    logger.info(f"Saved Joblib Model to:        {saved_model_path.resolve()}")
    logger.info(f"Saved Pickle Model to:        {saved_pickle_path.resolve()}")
    logger.info(f"Saved MLflow Model Package to:{mlflow_model_dir.resolve()}")
    logger.info(f"Saved Model Metrics Gate to:  {(MODELS_DIR / 'model_metrics.json').resolve()}")
    logger.info("=======================================================")

    # Write summary comparison JSON
    comparison_summary = {
        "selected_model": best_overall_name,
        "selection_criteria": "Clinical Recall priority (60%) + Discriminative ROC-AUC (40%)",
        "models": models_comparison,
    }
    summary_path = REPORTS_DIR / "model_comparison.json"
    with open(summary_path, "w") as f:
        json.dump(comparison_summary, f, indent=2)
    logger.info(f"Saved model comparison report to {summary_path}")

    return models_comparison, trained_pipelines, roc_plot_data


if __name__ == "__main__":
    train_and_track()
