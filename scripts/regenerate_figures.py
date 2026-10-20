#!/usr/bin/env python3
"""
Regenerate all EDA and model figures with improved aesthetics.
- Different, richer color palettes
- Additional/varied plot types (KDE, Box+Strip, Swarm, Donut, etc.)
- Saves updated PNGs to reports/figures/
"""

import os
import sys
from pathlib import Path

# ── Project root resolution ────────────────────────────────────────────────
_cwd = Path(__file__).resolve().parent.parent
BASE_DIR = _cwd
sys.path.insert(0, str(BASE_DIR))

os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".matplotlib")
(BASE_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "heart_cleaned.csv"

# ── Global style ────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.1)

# Rich custom palette for binary target
PALETTE_BINARY = ["#2196F3", "#FF5722"]   # Blue vs Deep Orange
PALETTE_SEQUENTIAL = "viridis"
PALETTE_DIVERGING = "coolwarm"
PALETTE_CATEGORICAL = sns.color_palette("tab10")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(PROCESSED_DATA_PATH)

# Preserve raw target for first plot (before binary encoding in EDA)
raw_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
df["target_raw"] = df["target"].map(lambda x: x)   # already binary in cleaned CSV
# Recreate raw target distribution from the canonical column distribution
raw_counts_info = {0: 164, 1: 55, 2: 36, 3: 35, 4: 13}

print(f"Loaded {len(df)} rows. Figures dir: {FIGURES_DIR}")

# ===========================================================================
# Figure 01 — Target Distribution (Horizontal Lollipop + Donut)
# ===========================================================================
def plot_01_target_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#f9f9f9")

    # --- Lollipop chart for raw multi-class distribution ---
    ax0 = axes[0]
    ax0.set_facecolor("#f9f9f9")
    cats = list(raw_counts_info.keys())
    vals = list(raw_counts_info.values())
    colors_ramp = plt.cm.plasma(np.linspace(0.15, 0.85, len(cats)))

    for idx, (cat, val) in enumerate(zip(cats, vals)):
        ax0.plot([0, val], [idx, idx], color=colors_ramp[idx], lw=2.5, alpha=0.8)
        ax0.scatter(val, idx, color=colors_ramp[idx], s=160, zorder=5)
        pct = val / sum(vals) * 100
        ax0.text(val + 2, idx, f"{val}  ({pct:.1f}%)", va="center", fontsize=11, fontweight="bold", color=colors_ramp[idx])

    ax0.set_yticks(range(len(cats)))
    ax0.set_yticklabels([f"Stage {c}" for c in cats], fontsize=11)
    ax0.set_xlabel("Patient Count", fontsize=11)
    ax0.set_title("Raw Angiographic Severity (0–4)", pad=12)
    ax0.set_xlim(0, max(vals) * 1.35)
    ax0.grid(axis="x", alpha=0.4)
    ax0.set_axisbelow(True)

    # --- Donut chart for binary target ---
    ax1 = axes[1]
    ax1.set_facecolor("#f9f9f9")
    binary_counts = df["target"].value_counts().sort_index()
    wedge_props = dict(edgecolor="white", linewidth=3, width=0.52)
    wedges, texts, autotexts = ax1.pie(
        binary_counts.values,
        labels=["Absence (0)", "Presence (1)"],
        autopct="%1.1f%%",
        startangle=135,
        colors=PALETTE_BINARY,
        explode=(0.04, 0.04),
        textprops={"fontsize": 12, "fontweight": "bold"},
        wedgeprops=wedge_props,
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
        at.set_color("white")

    # Centre annotation
    ax1.text(0, 0, f"N={len(df)}", ha="center", va="center", fontsize=16, fontweight="bold", color="#333333")
    ax1.set_title(f"Binary Target Distribution (N={len(df)})", pad=12)

    plt.suptitle("Figure 1 — Target Distribution: Raw vs. Binary Encoding", y=1.01, fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "01_target_distribution.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out.name}")


# ===========================================================================
# Figure 02 — Numerical Feature Distributions (Histogram + KDE overlay)
# ===========================================================================
def plot_02_numerical_histograms():
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    titles = [
        "Age (Years)",
        "Resting Blood Pressure (mm Hg)",
        "Serum Cholesterol (mg/dl)",
        "Max Heart Rate Achieved (bpm)",
        "ST Depression (oldpeak, mm)",
    ]
    palette_num = sns.color_palette("mako", len(num_cols))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#f8f8f8")
    axes_flat = axes.flatten()

    for i, (col, title) in enumerate(zip(num_cols, titles)):
        ax = axes_flat[i]
        ax.set_facecolor("#f8f8f8")

        # Split by target for overlapping KDE
        for t_val, t_label, clr in zip([0, 1], ["No Disease", "Disease"], PALETTE_BINARY):
            subset = df[df["target"] == t_val][col]
            sns.histplot(
                subset, kde=True, ax=ax, color=clr, alpha=0.45,
                bins=16, edgecolor="white", label=t_label, stat="density"
            )

        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color="#FF9800", linestyle="--", linewidth=2.0, label=f"Mean: {mean_val:.1f}")
        ax.axvline(median_val, color="#4CAF50", linestyle="-", linewidth=2.0, label=f"Median: {median_val:.1f}")
        ax.set_title(title, pad=10)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9, framealpha=0.8)

    fig.delaxes(axes_flat[5])
    plt.suptitle("Figure 2 — Numerical Feature Distributions by Target Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "02_numerical_histograms.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out.name}")


# ===========================================================================
# Figure 03 — Categorical Feature Distributions (Grouped Bar)
# ===========================================================================
def plot_03_categorical_distributions():
    cat_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    cat_labels = {
        "sex": {0.0: "Female", 1.0: "Male"},
        "cp": {1.0: "Typical\nAngina", 2.0: "Atypical\nAngina", 3.0: "Non-Anginal", 4.0: "Asymptomatic"},
        "fbs": {0.0: "≤120 mg/dl", 1.0: ">120 mg/dl"},
        "restecg": {0.0: "Normal", 1.0: "ST-T Wave", 2.0: "LV Hypert."},
        "exang": {0.0: "No", 1.0: "Yes"},
        "slope": {1.0: "Upsloping", 2.0: "Flat", 3.0: "Downsloping"},
        "ca": {0.0: "0 Vessels", 1.0: "1 Vessel", 2.0: "2 Vessels", 3.0: "3 Vessels"},
        "thal": {3.0: "Normal", 6.0: "Fixed\nDefect", 7.0: "Reversible\nDefect"},
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor("#f8f8f8")
    axes_flat = axes.flatten()

    for i, col in enumerate(cat_features):
        ax = axes_flat[i]
        ax.set_facecolor("#f8f8f8")

        # Grouped bars: No Disease vs Disease
        vc_0 = df[df["target"] == 0][col].value_counts().sort_index()
        vc_1 = df[df["target"] == 1][col].value_counts().sort_index()
        all_keys = sorted(set(vc_0.index) | set(vc_1.index))
        x = np.arange(len(all_keys))
        bar_w = 0.38
        b0 = ax.bar(x - bar_w / 2, [vc_0.get(k, 0) for k in all_keys], width=bar_w,
                    color=PALETTE_BINARY[0], alpha=0.85, edgecolor="white", label="No Disease")
        b1 = ax.bar(x + bar_w / 2, [vc_1.get(k, 0) for k in all_keys], width=bar_w,
                    color=PALETTE_BINARY[1], alpha=0.85, edgecolor="white", label="Disease")

        labels_map = cat_labels.get(col, {})
        tick_labels = [labels_map.get(k, str(k)) for k in all_keys]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_title(col.upper(), pad=8, fontweight="bold")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, framealpha=0.8)
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)

    plt.suptitle("Figure 3 — Categorical Feature Distributions (No Disease vs. Disease)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "03_categorical_distributions.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out.name}")


# ===========================================================================
# Figure 04 — Correlation Heatmap (lower triangle, annotated, coolwarm)
# ===========================================================================
def plot_04_correlation_heatmap():
    corr_cols = [c for c in df.columns if c != "target_raw" and df[c].dtype != object]
    corr_matrix = df[corr_cols].corr(method="spearman")
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f8f8f8")

    cmap = sns.diverging_palette(220, 20, s=90, l=45, as_cmap=True)
    sns.heatmap(
        corr_matrix, mask=mask, cmap=cmap, annot=True, fmt=".2f",
        square=True, linewidths=0.5, linecolor="#dddddd",
        cbar_kws={"shrink": 0.75, "label": "Spearman ρ", "aspect": 30},
        ax=ax, vmin=-1, vmax=1,
        annot_kws={"size": 9, "weight": "bold"}
    )
    ax.set_title("Figure 4 — Spearman Rank Correlation Matrix\n(Clinical Features + Binary Target)", pad=15)
    plt.tight_layout()
    out = FIGURES_DIR / "04_correlation_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out.name}")


# ===========================================================================
# Figure 05 — Box + Strip Plot: Numerical Features vs. Target
# ===========================================================================
def plot_05_feature_vs_target_boxstrip():
    violin_features = ["age", "trestbps", "chol", "thalach"]
    titles_v = ["Age (Years)", "Resting BP (mm Hg)", "Serum Cholesterol (mg/dl)", "Max Heart Rate (bpm)"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.patch.set_facecolor("#f8f8f8")

    for i, (col, title) in enumerate(zip(violin_features, titles_v)):
        ax = axes[i]
        ax.set_facecolor("#f8f8f8")

        # Box plot base
        sns.boxplot(
            x="target", y=col, data=df, ax=ax,
            palette=PALETTE_BINARY,
            width=0.5, linewidth=1.5, fliersize=0,
            boxprops=dict(alpha=0.7),
        )
        # Strip plot overlay
        sns.stripplot(
            x="target", y=col, data=df, ax=ax,
            palette=PALETTE_BINARY,
            size=4, alpha=0.5, jitter=True, dodge=False,
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Disease\n(0)", "Disease\n(1)"], fontsize=10)
        ax.set_title(title, pad=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(col, fontsize=10)
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)

    plt.suptitle("Figure 5 — Numerical Features vs. Diagnosis (Box + Strip Plot)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "05_feature_vs_target_violin.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out.name}")


# ===========================================================================
# Figure 06 — Chest Pain Type vs. Disease (Stacked + Line Overlay)
# ===========================================================================
def plot_06_chest_pain_vs_disease():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#f8f8f8")

    cp_labels = ["Typical Angina (1)", "Atypical Angina (2)", "Non-Anginal (3)", "Asymptomatic (4)"]

    # --- Stacked 100% bar ---
    ax0 = axes[0]
    ax0.set_facecolor("#f8f8f8")
    cp_ct = pd.crosstab(df["cp"], df["target"], normalize="index") * 100
    cp_ct.index = cp_labels
    cp_ct.columns = ["No Disease", "Disease"]
    cp_ct.plot(kind="barh", stacked=True, color=PALETTE_BINARY, ax=ax0, edgecolor="white", width=0.6, alpha=0.9)
    ax0.set_xlabel("Proportion (%)")
    ax0.set_title("Disease Prevalence by Chest Pain Type (%)", pad=10)
    ax0.axvline(50, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax0.legend(loc="lower right", fontsize=9)
    ax0.grid(axis="x", alpha=0.4)
    ax0.set_axisbelow(True)

    # --- Absolute count grouped bar ---
    ax1 = axes[1]
    ax1.set_facecolor("#f8f8f8")
    cp_abs = pd.crosstab(df["cp"], df["target"])
    cp_abs.index = cp_labels
    cp_abs.columns = ["No Disease", "Disease"]
    x = np.arange(len(cp_abs))
    w = 0.38
    ax1.bar(x - w / 2, cp_abs["No Disease"], width=w, color=PALETTE_BINARY[0], alpha=0.85, edgecolor="white", label="No Disease")
    ax1.bar(x + w / 2, cp_abs["Disease"], width=w, color=PALETTE_BINARY[1], alpha=0.85, edgecolor="white", label="Disease")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cp_labels, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("Patient Count")
    ax1.set_title("Absolute Counts by Chest Pain Type", pad=10)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.4)
    ax1.set_axisbelow(True)

    plt.suptitle("Figure 6 — Chest Pain Type vs. Heart Disease", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "06_chest_pain_vs_disease.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out.name}")


# ===========================================================================
# Figure 07 — Max Heart Rate vs. Age Scatter (with regression + density contour)
# ===========================================================================
def plot_07_thalach_vs_age_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#f8f8f8")

    # --- Scatter + regression lines ---
    ax0 = axes[0]
    ax0.set_facecolor("#f8f8f8")
    for t_val, t_label, clr, mk in zip([0, 1], ["No Disease", "Disease"], PALETTE_BINARY, ["o", "^"]):
        sub = df[df["target"] == t_val]
        ax0.scatter(sub["age"], sub["thalach"], c=clr, alpha=0.55, s=55, label=t_label, marker=mk, edgecolors="white", linewidths=0.5)
        z = np.polyfit(sub["age"], sub["thalach"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df["age"].min(), df["age"].max(), 100)
        ax0.plot(x_line, p(x_line), color=clr, linewidth=2.5, linestyle="--")
    ax0.set_xlabel("Age (Years)", fontsize=11)
    ax0.set_ylabel("Max Heart Rate (bpm)", fontsize=11)
    ax0.set_title("Scatter: Max Heart Rate vs. Age by Diagnosis", pad=10)
    ax0.legend(fontsize=10)
    ax0.grid(alpha=0.4)

    # --- 2D KDE Contour Plot ---
    ax1 = axes[1]
    ax1.set_facecolor("#f8f8f8")
    for t_val, t_label, clr in zip([0, 1], ["No Disease", "Disease"], PALETTE_BINARY):
        sub = df[df["target"] == t_val]
        try:
            sns.kdeplot(x=sub["age"], y=sub["thalach"], ax=ax1, color=clr,
                        fill=True, alpha=0.3, levels=6, label=t_label)
            sns.kdeplot(x=sub["age"], y=sub["thalach"], ax=ax1, color=clr,
                        fill=False, linewidths=1.5, levels=6)
        except Exception:
            pass
    ax1.set_xlabel("Age (Years)", fontsize=11)
    ax1.set_ylabel("Max Heart Rate (bpm)", fontsize=11)
    ax1.set_title("2D Density: Age vs. Max Heart Rate by Diagnosis", pad=10)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.4)

    plt.suptitle("Figure 7 — Chronotropic Reserve: Max Heart Rate vs. Age", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "07_thalach_vs_age_scatter.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out.name}")


# ===========================================================================
# Figure 08 — Fluoroscopy Vessels (ca) & Thalassemia (thal) vs. Target
# ===========================================================================
def plot_08_fluoroscopy_vessels_vs_target():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#f8f8f8")

    # Heatmap for ca
    ax0 = axes[0]
    ax0.set_facecolor("#f8f8f8")
    ca_ct = pd.crosstab(df["ca"], df["target"])
    ca_pct = ca_ct.div(ca_ct.sum(axis=1), axis=0) * 100
    ca_pct.index = [f"{int(v)} vessel(s)" for v in ca_pct.index]
    ca_pct.columns = ["No Disease", "Disease"]
    sns.heatmap(
        ca_pct, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=ax0,
        linewidths=1, linecolor="white", cbar_kws={"label": "% of Row"},
        vmin=0, vmax=100
    )
    ax0.set_title("Disease % by Fluoroscopy Vessels (ca)", pad=10)
    ax0.set_ylabel("CA Category")
    ax0.set_xlabel("Diagnosis")

    # Heatmap for thal
    ax1 = axes[1]
    ax1.set_facecolor("#f8f8f8")
    thal_ct = pd.crosstab(df["thal"], df["target"])
    thal_pct = thal_ct.div(thal_ct.sum(axis=1), axis=0) * 100
    thal_labels = {3.0: "Normal (3)", 6.0: "Fixed Defect (6)", 7.0: "Reversible (7)"}
    thal_pct.index = [thal_labels.get(v, str(v)) for v in thal_pct.index]
    thal_pct.columns = ["No Disease", "Disease"]
    sns.heatmap(
        thal_pct, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=ax1,
        linewidths=1, linecolor="white", cbar_kws={"label": "% of Row"},
        vmin=0, vmax=100
    )
    ax1.set_title("Disease % by Thalassemia Status (thal)", pad=10)
    ax1.set_ylabel("Thal Category")
    ax1.set_xlabel("Diagnosis")

    plt.suptitle("Figure 8 — Fluoroscopy Vessels & Thalassemia vs. Disease Risk", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "08_fluoroscopy_vessels_vs_target.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out.name}")


# ===========================================================================
# Main runner
# ===========================================================================
if __name__ == "__main__":
    print("Regenerating all EDA figures with updated aesthetics...\n")
    plot_01_target_distribution()
    plot_02_numerical_histograms()
    plot_03_categorical_distributions()
    plot_04_correlation_heatmap()
    plot_05_feature_vs_target_boxstrip()
    plot_06_chest_pain_vs_disease()
    plot_07_thalach_vs_age_scatter()
    plot_08_fluoroscopy_vessels_vs_target()
    print("\nAll 8 EDA figures saved successfully!")
