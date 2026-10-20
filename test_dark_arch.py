import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
OUTPUT_DIR = Path("reports/figures")

def generate_dark_architecture():
    # 20 x 11.8 canvas in 300 DPI gives ultra-crisp resolution
    fig, ax = plt.subplots(figsize=(20, 11.8), facecolor="#090d16")
    ax.set_facecolor("#090d16")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header section
    # Platform badge
    badge_rect = patches.FancyBboxPatch(
        (0.36, 0.958), 0.28, 0.024,
        boxstyle="round,pad=0.002,rounding_size=0.01",
        facecolor="#064e3b", edgecolor="#10b981", linewidth=1.2
    )
    ax.add_patch(badge_rect)
    ax.text(0.5, 0.970, "ENTERPRISE MLOPS SYSTEM ARCHITECTURE",
            ha="center", va="center", fontsize=9.5, weight="bold", color="#34d399")

    # Title & Subtitle
    ax.text(0.5, 0.932, "HeartGuard: End-to-End Production ML Platform",
            ha="center", va="center", fontsize=22, weight="bold", color="#f8fafc")
    ax.text(0.5, 0.903,
            "Clinical Risk Stratification  •  MLflow Governance  •  Automated CI/CD Quality Gate  •  Kubernetes Orchestration & Drift Telemetry",
            ha="center", va="center", fontsize=10.5, color="#94a3b8")

    # Layout dimensions
    # col_width = 0.211, gap = 0.038 -> total = 4*0.211 + 3*0.038 = 0.844 + 0.114 = 0.958
    # x_start = (1.0 - 0.958) / 2 = 0.021
    col_width = 0.211
    gap = 0.038
    x_start = 0.021
    container_y = 0.106
    container_h = 0.774

    card_w = col_width - 0.020
    card_h = 0.203
    card_y_positions = [0.608, 0.364, 0.120]

    # Define the 4 Tiers
    tiers = [
        {
            "tag": "TIER 01: INGESTION & PREP",
            "title": "Data & Preprocessing",
            "accent": "#38bdf8",
            "border": "#0284c7",
            "bg": "#0c1929",
            "card_bg": "#111f36",
            "x": x_start + 0 * (col_width + gap),
            "cards": [
                {
                    "title": "UCI Cleveland Cohort",
                    "lines": [
                        "• Canonical 303 patient clinical records",
                        "• 13 physiological features + cardiac target",
                        "• Strict Pydantic schema & integrity validation"
                    ],
                    "badges": ["UCI ML API", "Pandas 2.2", "Pydantic"]
                },
                {
                    "title": "Data Hygiene & Cleaning",
                    "lines": [
                        "• Ca / Thal missingness handling (median/mode)",
                        "• Target binarisation (0: Healthy vs 1..4: Disease)",
                        "• Stratified 80/20 train/test cohort split"
                    ],
                    "badges": ["NumPy", "StratifiedSplit", "Zero-Leakage"]
                },
                {
                    "title": "Feature Transformer Pipeline",
                    "lines": [
                        "• SimpleImputer (median for numeric, mode for cat)",
                        "• StandardScaler (chol, trestbps, thalach, age)",
                        "• OneHotEncoder (cp, restecg, slope, thal)"
                    ],
                    "badges": ["ColumnTransformer", "Scikit-Learn"]
                }
            ]
        },
        {
            "tag": "TIER 02: TRAINING & GOVERNANCE",
            "title": "Training & MLflow Tracking",
            "accent": "#34d399",
            "border": "#059669",
            "bg": "#081d18",
            "card_bg": "#0e2a22",
            "x": x_start + 1 * (col_width + gap),
            "cards": [
                {
                    "title": "Dual Model Candidates",
                    "lines": [
                        "• Regularized Logistic Regression (C=5.0)",
                        "• Random Forest Classifier (Depth=5, 100 Trees)",
                        "• Full pipeline encapsulation to curb leakage"
                    ],
                    "badges": ["Scikit-Learn", "LogisticReg", "RandomForest"]
                },
                {
                    "title": "Stratified 5-Fold CV Tuning",
                    "lines": [
                        "• GridSearchCV cross-validation on train fold",
                        "• Optimized for ROC-AUC & Clinical Sensitivity",
                        "• Deterministic reproducibility (seed = 42)"
                    ],
                    "badges": ["GridSearchCV", "StratifiedKFold", "ROC-AUC"]
                },
                {
                    "title": "MLflow Experiment Tracking",
                    "lines": [
                        "• Telemetry: hyperparams, metrics, ROC & PR curves",
                        "• Confusion matrix artifacts & model serialization",
                        "• Winning model: 0.9632 AUC, 89.29% Recall"
                    ],
                    "badges": ["MLflow 2.15", "Artifact Store", "Run Registry"]
                }
            ]
        },
        {
            "tag": "TIER 03: PACKAGING & CI/CD",
            "title": "Packaging & CI/CD Gate",
            "accent": "#fbbf24",
            "border": "#d97706",
            "bg": "#1c1608",
            "card_bg": "#291f0c",
            "x": x_start + 2 * (col_width + gap),
            "cards": [
                {
                    "title": "Model Packaging & Lockfiles",
                    "lines": [
                        "• Production artifact: models/model_pipeline.joblib",
                        "• Feature schema metadata & inference contract",
                        "• Pinned requirements.txt & environment locks"
                    ],
                    "badges": ["Joblib", "Schema Guard", "Reproducible"]
                },
                {
                    "title": "Automated Pytest Gate",
                    "lines": [
                        "• 29 Unit & Integration Tests (100% pass)",
                        "• Data schema, pipeline, & API contract tests",
                        "• Sub-6ms latency benchmark verification"
                    ],
                    "badges": ["Pytest 8.3", "Test Coverage", "Quality Gate"]
                },
                {
                    "title": "GitHub Actions CI/CD",
                    "lines": [
                        "• Ruff & Black code style enforcement",
                        "• Automated training & artifact verification",
                        "• Multi-step artifact logging & container check"
                    ],
                    "badges": ["GitHub Actions", "CI Workflows", "Lint / Test"]
                }
            ]
        },
        {
            "tag": "TIER 04: SERVING & OBSERVABILITY",
            "title": "Serving & Observability",
            "accent": "#c084fc",
            "border": "#9333ea",
            "bg": "#190d26",
            "card_bg": "#25143a",
            "x": x_start + 3 * (col_width + gap),
            "cards": [
                {
                    "title": "FastAPI Inference REST API",
                    "lines": [
                        "• Endpoints: /predict, /health, /metrics, /drift",
                        "• Pydantic clinical input contract validation",
                        "• Sub-6ms inference • Probability & Risk Score"
                    ],
                    "badges": ["FastAPI", "Uvicorn", "Pydantic v2"]
                },
                {
                    "title": "Container & K8s Orchestration",
                    "lines": [
                        "• Non-root secure Docker container (UID 10001)",
                        "• 3-replica K8s Deployment & LoadBalancer Service",
                        "• Horizontal Pod Autoscaler (HPA) & Helm Chart"
                    ],
                    "badges": ["Docker", "Kubernetes", "Helm 3", "HPA"]
                },
                {
                    "title": "Prometheus & Drift Engine",
                    "lines": [
                        "• Real-time request latency & Prometheus exporter",
                        "• Kolmogorov-Smirnov & Population Stability Index",
                        "• Real-time drift detection & Grafana telemetry"
                    ],
                    "badges": ["Prometheus", "Grafana", "KS & PSI Drift"]
                }
            ]
        }
    ]

    for tier in tiers:
        x = tier["x"]
        accent = tier["accent"]
        border = tier["border"]
        bg = tier["bg"]
        card_bg = tier["card_bg"]

        # Outer Tier Container
        container = patches.FancyBboxPatch(
            (x, container_y), col_width, container_h,
            boxstyle="round,pad=0.005,rounding_size=0.016",
            facecolor=bg, edgecolor=border, linewidth=1.8
        )
        ax.add_patch(container)

        # Tier Top Pill Tag
        tag_w = 0.146
        tag_h = 0.020
        tag_box = patches.FancyBboxPatch(
            (x + (col_width - tag_w)/2, 0.846), tag_w, tag_h,
            boxstyle="round,pad=0.002,rounding_size=0.008",
            facecolor="#0f172a", edgecolor=accent, linewidth=1.0
        )
        ax.add_patch(tag_box)
        ax.text(x + col_width/2, 0.856, tier["tag"],
                ha="center", va="center", fontsize=7.2, weight="bold", color=accent)

        # Tier Main Title
        ax.text(x + col_width/2, 0.824, tier["title"],
                ha="center", va="center", fontsize=12.2, weight="bold", color="#f8fafc")

        # 3 Component Cards
        for idx, (cy, card) in enumerate(zip(card_y_positions, tier["cards"])):
            cx = x + 0.010
            card_rect = patches.FancyBboxPatch(
                (cx, cy), card_w, card_h,
                boxstyle="round,pad=0.003,rounding_size=0.012",
                facecolor=card_bg, edgecolor="#334155", linewidth=1.1
            )
            ax.add_patch(card_rect)

            # Card Header with small accent bullet
            ax.plot([cx + 0.010], [cy + card_h - 0.024], marker="o", markersize=4.5, color=accent)
            ax.text(cx + 0.018, cy + card_h - 0.024, card["title"],
                    ha="left", va="center", fontsize=10.2, weight="bold", color="#f1f5f9")

            # Separator line
            ax.plot([cx + 0.010, cx + card_w - 0.010], [cy + card_h - 0.038, cy + card_h - 0.038],
                    color="#334155", linewidth=0.8, alpha=0.7)

            # Bullet lines
            line_y = cy + card_h - 0.058
            for line in card["lines"]:
                ax.text(cx + 0.010, line_y, line,
                        ha="left", va="center", fontsize=8.0, color="#cbd5e1")
                line_y -= 0.030

            # Tech badges along card bottom
            badge_x = cx + 0.010
            badge_y = cy + 0.022
            for badge in card["badges"]:
                b_len = len(badge) * 0.0039 + 0.011
                b_rect = patches.FancyBboxPatch(
                    (badge_x, badge_y - 0.009), b_len, 0.018,
                    boxstyle="round,pad=0.001,rounding_size=0.006",
                    facecolor="#0f172a", edgecolor=accent, linewidth=0.8, alpha=0.9
                )
                ax.add_patch(b_rect)
                ax.text(badge_x + b_len/2, badge_y, badge,
                        ha="center", va="center", fontsize=6.8, weight="bold", color=accent)
                badge_x += b_len + 0.005

            # Vertical connector arrows between cards inside the tier
            if idx < 2:
                next_cy = card_y_positions[idx + 1]
                arrow_y_start = cy - 0.002
                arrow_y_end = next_cy + card_h + 0.002
                ax.annotate(
                    "",
                    xy=(x + col_width/2, arrow_y_end),
                    xytext=(x + col_width/2, arrow_y_start),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=accent,
                        lw=1.5,
                        mutation_scale=10,
                        alpha=0.65
                    )
                )

    # Inter-Tier Connectors (Horizontal flow channels between columns)
    connectors = [
        (tiers[0]["x"] + col_width, tiers[1]["x"], "Clean Data\n& Splits", "#38bdf8"),
        (tiers[1]["x"] + col_width, tiers[2]["x"], "Tuned Model\n& Artifacts", "#34d399"),
        (tiers[2]["x"] + col_width, tiers[3]["x"], "Validated\nDocker Image", "#fbbf24"),
    ]

    for x1, x2, flow_label, color in connectors:
        mid_x = (x1 + x2) / 2
        # Arrow across the gap at mid height y=0.48
        ax.annotate(
            "",
            xy=(x2 - 0.003, 0.48),
            xytext=(x1 + 0.003, 0.48),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=2.5,
                mutation_scale=14,
            )
        )
        # Flow label pill cleanly placed in the gap without touching containers
        ax.text(mid_x, 0.525, flow_label,
                ha="center", va="center", fontsize=6.0, weight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#090d16", edgecolor=color, lw=0.9))

    # Bottom Enterprise Compliance & Security Banner
    footer_rect = patches.FancyBboxPatch(
        (0.021, 0.020), 0.958, 0.058,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor="#0f172a", edgecolor="#334155", linewidth=1.2
    )
    ax.add_patch(footer_rect)

    # 5 Pillars in Footer (using clean geometric bullets)
    pillars = [
        ("✦  Zero-Leakage Guarantee", "Fitted strictly on train folds", "#38bdf8"),
        ("✦  Non-Root Security", "UID 10001 unprivileged user", "#34d399"),
        ("✦  High-Sensitivity Recall", "89.29% clinical risk capture", "#fbbf24"),
        ("✦  Kubernetes Resiliency", "3 Replicas + Ingress + HPA", "#a855f7"),
        ("✦  Real-Time Observability", "Prometheus & KS Drift Engine", "#f43f5e")
    ]

    p_w = 0.958 / len(pillars)
    for i, (title, desc, color) in enumerate(pillars):
        px = 0.021 + i * p_w + p_w / 2
        ax.text(px, 0.057, title, ha="center", va="center", fontsize=9.0, weight="bold", color=color)
        ax.text(px, 0.036, desc, ha="center", va="center", fontsize=7.4, color="#94a3b8")
        if i < len(pillars) - 1:
            ax.plot([0.021 + (i+1)*p_w, 0.021 + (i+1)*p_w], [0.026, 0.072], color="#334155", lw=1.0)

    plt.tight_layout()
    out_file = OUTPUT_DIR / "architecture_diagram.png"
    plt.savefig(out_file, dpi=300, facecolor="#090d16", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print("Done! Saved to:", out_file)

generate_dark_architecture()
