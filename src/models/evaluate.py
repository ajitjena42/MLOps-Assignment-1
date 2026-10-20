"""Model evaluation utilities, metrics calculation, and diagnostic plotting."""

import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPL_DIR = PROJECT_ROOT / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict[str, Any]:
    """Calculate standard and healthcare-specific diagnostic performance metrics.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_proba: Predicted probability of positive class (Class 1).
        
    Returns:
        Dictionary of performance metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "false_negative_rate": float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0,
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path
) -> Path:
    """Plot publication-quality confusion matrix highlighting clinical implications.
    
    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted labels.
        model_name: Display name of the model.
        output_path: Destination image path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    annot = np.array([
        [f"TN\n{tn}\n({tn/(tn+fp)*100:.1f}%)", f"FP (Type I)\n{fp}\n({fp/(tn+fp)*100:.1f}%)"],
        [f"FN (Type II)\n{fn}\n({fn/(fn+tp)*100:.1f}%)", f"TP\n{tp}\n({tp/(fn+tp)*100:.1f}%)"]
    ])

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Absence (0)", "Disease Risk (1)"],
        yticklabels=["Absence (0)", "Disease Risk (1)"],
        ax=ax,
        annot_kws={"fontsize": 11, "weight": "bold"}
    )

    ax.set_title(f"Confusion Matrix - {model_name}\n(Held-Out Test Set, N={len(y_true)})", pad=12)
    ax.set_xlabel("Predicted Diagnosis")
    ax.set_ylabel("True Angiographic Diagnosis")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_roc_curves(
    models_roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    output_path: Path
) -> Path:
    """Plot overlaid ROC curves comparing multiple candidate models.
    
    Args:
        models_roc_data: Dictionary mapping model_name -> (fpr, tpr, roc_auc).
        output_path: Destination image path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2b5c8f", "#d95f02", "#2ca02c", "#7570b3"]
    for idx, (name, (fpr, tpr, score)) in enumerate(models_roc_data.items()):
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUC = {score:.3f})")

    ax.plot([0, 1], [0, 1], color="#888888", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    model_name: str,
    output_path: Path,
    title: str = "Feature Importance"
) -> Path:
    """Plot sorted top feature importances or model coefficients."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.argsort(np.abs(importances))[-15:]  # top 15

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(indices)), importances[indices], color="#2b5c8f", edgecolor="black")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Magnitude")
    ax.set_title(f"{title} - {model_name}", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
